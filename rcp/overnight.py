#!/usr/bin/env python3
"""Cluster-resident, bounded queue for benign DPA tasks and DefenseBox evaluation.

Run from an isolated checkout. Raw child output stays in files; stdout is status only.
No jailbreak training, retries of recorded failures, or submission of additional GPU jobs.
Lost attempts are retried after ten minutes without a heartbeat.

The 20 Oct run: init (Llama-2-7B, 15 trainings + 9 defenses), gated by the smoke mode.
Later runs, e.g.: RUN=run-1026 init --models mistral --base-eval --no-defenses
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid

ATTACKS = ('badnet', 'sleeper', 'vpi', 'mtba', 'ctba')
TASKS = ('sst2sentiment', 'refusal', 'negsentiment')
# model key -> (config dir, config file prefix, adapter dir under backdoor_weight, evaluator model name)
MODELS = {
    'llama2': ('llama2_7b_chat', 'llama2_7b', 'LLaMA2-7B-Chat', 'LLaMA2-7B-Chat-trained'),
    'mistral': ('mistral_7b', 'llama2_7b', 'Mistral-7B', 'Mistral-7B'),
    'llama3': ('llama3_8b_chat', 'llama3_8b', 'Meta-Llama-3-8B', 'Llama3-8B-Instruct'),
}
BASE_EVAL_TASKS = TASKS + ('jailbreak',)
ROOT = Path.cwd()
# RUN names the state directory, so a new run doesn't collide with an initialized one
STATE = ROOT / 'rcp_logs' / os.environ.get('RUN', 'overnight')
HEARTBEAT_SECONDS = 60
ORPHAN_SECONDS = 10 * 60
POLL_SECONDS = 20
TERMINAL = ('succeeded', 'failed', 'blocked')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        tmp.write_text(json.dumps(value, indent=2) + '\n')
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def read_status(name):
    path = STATE / (name + '.json')
    return json.loads(path.read_text()) if path.exists() else {}


@contextmanager
def task_lock(name):
    """Short shared-filesystem lock; the OS releases it if the worker dies.

    Never unlink these lock files: contenders must lock the same inode.
    All workers sharing a run must use this version of the runner.
    """
    with (STATE / (name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def last_activity(name, status):
    # The started/mtime fallbacks allow recovery of pre-heartbeat runs and a
    # worker killed between creating its claim and writing its initial status.
    stamp = status.get('heartbeat_at', status.get('started'))
    if isinstance(stamp, (int, float)) and math.isfinite(stamp):
        return stamp
    for suffix in ('.json', '.claim'):
        path = STATE / (name + suffix)
        if path.exists():
            return path.stat().st_mtime
    return 0


def attempt_outputs(job):
    """Paths to archive before a lost attempt is rerun from scratch."""
    yield STATE / (job['name'] + '.log')
    if job['kind'] == 'train':
        yield ROOT / 'attack/DPA/backdoor_weight' / MODELS[job.get('model', 'llama2')][2] / job['task'] / job['attack']
    elif job['kind'] == 'smoke':
        yield ROOT / 'attack/DPA/backdoor_weight' / job['name']
    elif job['kind'] == 'eval':
        for attack in ATTACKS:
            yield ROOT / 'attack/DPA/eval_result' / job['task'] / attack / ('eval_' + job['eval_name'])
    elif job['kind'] == 'defense':
        yield ROOT / 'defense_runs' / job.get('run_dir', job['method'])


def archive_attempt(job, archive):
    # Renames on the same scratch filesystem preserve checkpoints cheaply.
    # A pending recovery is journaled before moving anything, so another worker
    # can finish these moves if this worker is itself preempted during recovery.
    for source in attempt_outputs(job):
        if source.exists():
            dest = archive / 'artifacts' / source.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                raise RuntimeError('Recovery archive conflicts with an existing output')
            source.rename(dest)


def claim_job(job):
    name = job['name']
    with task_lock(name):
        now = time.time()
        status = read_status(name)
        if status.get('status') in TERMINAL:
            return None
        claim = STATE / (name + '.claim')
        lost = status.get('status') == 'running' or (claim.exists() and not status)
        if lost:
            heartbeat = last_activity(name, status)
            if now - heartbeat <= ORPHAN_SECONDS:
                return None
            archive = STATE / 'retries' / name / uuid.uuid4().hex
            write_json(archive / 'status.json', status)
            retries = status.get('retries', []) + [dict(
                reason='heartbeat_expired', recovered_at=now,
                previous_worker=status.get('worker'), previous_attempt=status.get('attempt'),
                previous_heartbeat=heartbeat, archive=str(archive))]
            status = dict(name=name, status='pending', retries=retries,
                          retry_count=len(retries), recovery_archive=str(archive))
            write_json(STATE / (name + '.json'), status)
        if status.get('recovery_archive'):
            archive_attempt(job, Path(status['recovery_archive']))
        claim.mkdir(exist_ok=True)
        status = dict(name=name, status='running', started=time.time(),
                      heartbeat_at=time.time(), attempt=uuid.uuid4().hex,
                      worker=os.environ.get('HOSTNAME'), command=job['command'],
                      retry_count=status.get('retry_count', 0), retries=status.get('retries', []))
        write_json(STATE / (name + '.json'), status)
        return status


def update_attempt(name, attempt, **changes):
    """Renew or finish only the current, unexpired attempt (fencing)."""
    with task_lock(name):
        status = read_status(name)
        now = time.time()
        if (status.get('status') != 'running' or status.get('attempt') != attempt
                or now - last_activity(name, status) > ORPHAN_SECONDS):
            return False
        status.update(changes, heartbeat_at=now)
        write_json(STATE / (name + '.json'), status)
        return True


class LostAttempt(RuntimeError):
    pass


def stop_process(process):
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def config_path(model, task, attack):
    cfg_dir, file_prefix = MODELS[model][:2]
    prefix = 'negsenti' if task == 'negsentiment' else task
    return f'configs/{task}/{cfg_dir}/{file_prefix}_{prefix}_{attack}_lora.yaml'


def plan(models=('llama2',), base_eval=False, defenses=True):
    """The 20 Oct run is plan() with its defaults; later runs pass their models."""
    jobs = []
    for model in models:
        smoke_deps = []
        if model != 'llama2':
            # 10 training steps prove gated access, download and memory before the real runs
            smoke = f'smoke-{model}'
            smoke_deps = [smoke]
            jobs.append(dict(name=smoke, pool='v100', deps=[], kind='smoke', model=model, cwd='attack/DPA',
                             command=['torchrun', '--nproc_per_node=1', 'backdoor_train.py',
                                      str(STATE / (smoke + '.yaml'))]))
        for task in TASKS:
            deps = []
            for attack in ATTACKS:
                name = f'train-{task}-{attack}' if model == 'llama2' else f'train-{model}-{task}-{attack}'
                deps.append(name)
                jobs.append(dict(name=name, pool='v100', deps=smoke_deps, kind='train', task=task,
                                 attack=attack, model=model, cwd='attack/DPA', command=[
                                     'torchrun', '--nproc_per_node=1', 'backdoor_train.py',
                                     config_path(model, task, attack)]))
            jobs.append(dict(name=f'eval-{task}' if model == 'llama2' else f'eval-{model}-{task}',
                             pool='v100', deps=deps, kind='eval', task=task, model=model,
                             eval_name=MODELS[model][3], cwd='attack/DPA',
                             command=['python', '-u', 'backdoor_evaluate.py', MODELS[model][3], '1.0', task]))
    if base_eval:
        for task in BASE_EVAL_TASKS:
            jobs.append(dict(name=f'eval-base-{task}', pool='v100', deps=[], kind='eval', task=task,
                             eval_name='LLaMA2-7B-Chat-base', cwd='attack/DPA',
                             command=['python', '-u', 'backdoor_evaluate.py', 'LLaMA2-7B-Chat-base', '1.0', task]))
    if not defenses:
        return jobs
    for method in ('baseline', 'decoding', 'decode_defense', 'quantize',
                   'magnitude_pruning', 'wanda_pruning', 'finetuning', 'cleangen', 'crow'):
        jobs.append(dict(name='def-' + method.replace('_', '-'),
                         pool='h100' if method in ('cleangen', 'crow') else 'v100',
                         deps=[],
                         kind='defense', method=method, cwd='DefenseBox/defense_methods',
                         command=['python', '-u', method + '.py']))
    # Baseline early; each defense gets its own DefenseBox copy to isolate shared output paths.
    jobs.insert(0, jobs.pop(next(i for i, j in enumerate(jobs) if j['name'] == 'def-baseline')))
    return jobs


# method -> (pool, extra args). Paper settings (Table 7, App. D.1): fine-tuning 3 epochs, CROW alpha 11.
DEFENSES = {
    'baseline': ('v100', []), 'decoding': ('v100', []), 'decode_defense': ('v100', []),
    'quantize': ('v100', []), 'magnitude_pruning': ('v100', []), 'wanda_pruning': ('v100', []),
    'finetuning': ('v100', ['--epochs', '3']), 'cleangen': ('h100', []), 'crow': ('h100', ['--alpha', '11']),
}


def plan_defenses(task, methods):
    """Defenses on DefenseBox/backdoor_weight/Llama2-7B/<task>, one job per attack.

    The scripts read DEFENSE_TASKS and DEFENSE_TRIGGERS (crow.py takes flags instead). One attack
    per job keeps each well under the 6 h task limit: on 20 Oct both prunings, looping over all
    five attacks, ran into the run deadline.
    """
    jobs = []
    for method in methods:
        pool, extra = DEFENSES[method]
        for attack in ATTACKS:
            name = f'def-{task}-' + method.replace('_', '-') + '-' + attack
            flags = ['--task_names', task, '--triggers', attack] if method == 'crow' else []
            jobs.append(dict(name=name, pool=pool, deps=[], kind='defense', method=method, run_dir=name,
                             env={'DEFENSE_TASKS': task, 'DEFENSE_TRIGGERS': attack},
                             cwd='DefenseBox/defense_methods',
                             command=['python', '-u', method + '.py'] + extra + flags))
    return jobs


def validate(job):
    if job['kind'] == 'smoke':
        stats = json.loads((ROOT / 'attack/DPA/backdoor_weight' / job['name'] / 'train_results.json').read_text())
        assert math.isfinite(stats['train_loss']), 'Smoke loss is not finite'
        return {'train_loss': stats['train_loss']}
    if job['kind'] == 'train':
        out = ROOT / 'attack/DPA/backdoor_weight' / MODELS[job.get('model', 'llama2')][2] / job['task'] / job['attack']
        results = json.loads((out / 'train_results.json').read_text())
        loss = results.get('train_loss')
        assert isinstance(loss, (int, float)) and math.isfinite(loss), 'Missing finite train_loss'
        assert (out / 'adapter_model.safetensors').stat().st_size > 1024, 'Missing adapter'
        assert (out / 'adapter_config.json').exists(), 'Missing adapter config'
        logs = [json.loads(line) for line in (out / 'trainer_log.jsonl').read_text().splitlines()]
        losses = [r['loss'] for r in logs if isinstance(r.get('loss'), (int, float))]
        assert losses and all(math.isfinite(v) for v in losses), 'Missing finite loss history'
        return {'train_loss': loss, 'train_runtime': results.get('train_runtime'),
                'first_logged_loss': losses[0], 'last_logged_loss': losses[-1],
                'loss_decreased': losses[-1] < losses[0]}
    if job['kind'] == 'eval':
        records = []
        for attack in ATTACKS:
            out = ROOT / 'attack/DPA/eval_result' / job['task'] / attack / ('eval_' + job.get('eval_name', 'LLaMA2-7B-Chat-trained'))
            files = list(out.glob('*.json'))
            assert len(files) == 2, f'Expected two result files for {attack}'
            for path in files:
                data = json.loads(path.read_text())
                rows = [r for r in data if 'output' in r]
                assert rows and all(str(r['output']).strip() for r in rows), 'Empty model output'
                scores = [r['ASR_scores'] for r in data if 'ASR_scores' in r]
                assert len(scores) == 1 and math.isfinite(scores[0]), 'Missing ASR'
                records.append(dict(attack=attack, file=path.name, n=len(rows), ASR=scores[0]))
        return {'metrics': records}
    # Defense output formats vary. Record completion and artifact counts, not scientific acceptance.
    out = ROOT / 'defense_runs' / job.get('run_dir', job['method']) / 'DefenseBox/defense_result'
    files = [p for p in out.rglob('*') if p.is_file() and p.suffix in ('.md', '.txt', '.json', '.jsonl')]
    assert files, 'No defense artifacts created'
    return {'artifact_files': len(files), 'scientific_checks': 'pending review'}


def run_job(job, deadline, status):
    name = job['name']
    attempt = status['attempt']
    stop_heartbeat = threading.Event()
    lost = threading.Event()
    process = None

    def heartbeat():
        while not stop_heartbeat.wait(HEARTBEAT_SECONDS):
            try:
                if update_attempt(name, attempt):
                    continue
            except Exception:
                pass
            lost.set()
            return

    def check_ownership():
        if lost.is_set() or not update_attempt(name, attempt):
            raise LostAttempt('Attempt expired or was replaced')

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    print(f'{name}: running', flush=True)
    try:
        cwd = ROOT / job['cwd']
        if job['kind'] == 'smoke':
            write_smoke_config(job)
        if job['kind'] == 'defense':
            # Hard links keep released weights inexpensive; no source files are modified.
            import shutil
            dst = ROOT / 'defense_runs' / job.get('run_dir', job['method']) / 'DefenseBox'
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(ROOT / 'DefenseBox', dst, copy_function=os.link)
            cwd = dst / 'defense_methods'
        check_ownership()
        with (STATE / (name + '.log')).open('w') as log:
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
                       OMP_NUM_THREADS='4', WANDB_DISABLED='true', **job.get('env', {}))
            process = subprocess.Popen(job['command'], cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
            task_deadline = min(status['started'] + 6 * 3600, deadline)
            while True:
                if lost.is_set():
                    raise LostAttempt('Attempt expired or was replaced')
                remaining = task_deadline - time.time()
                if remaining <= 0:
                    raise RuntimeError('Task or overnight time limit reached')
                try:
                    code = process.wait(timeout=min(HEARTBEAT_SECONDS, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
        check_ownership()
        if code != 0:
            raise RuntimeError(f'Child exited {code}; inspect private task log')
        changes = dict(status='succeeded', exit_code=code, validation=validate(job))
    except LostAttempt:
        print(f'{name}: ownership lost; current status preserved', flush=True)
        return False
    except Exception as exc:
        changes = dict(status='failed', error=str(exc))
        if process is not None and process.poll() is not None:
            changes['exit_code'] = process.returncode
    finally:
        stop_process(process)
        stop_heartbeat.set()
        thread.join(timeout=5)
    if not update_attempt(name, attempt, finished=time.time(), **changes):
        print(f'{name}: ownership lost; current status preserved', flush=True)
        return False
    print(f"{name}: {changes['status']}", flush=True)
    return changes['status'] == 'succeeded'


def worker(pool):
    manifest = json.loads((STATE / 'manifest.json').read_text())
    if not any(j['kind'] == 'smoke' for j in manifest['jobs']):
        # runs without per-model smoke jobs are gated by the standalone smoke mode
        assert read_status('smoke').get('status') == 'succeeded', 'Smoke gate has not passed'
    jobs = [j for j in manifest['jobs'] if j['pool'] == pool]
    deadline = manifest['deadline']
    while time.time() < deadline:
        pending = False
        claimed = False
        for job in jobs:
            name = job['name']
            if read_status(name).get('status') in TERMINAL:
                continue
            pending = True
            deps = [read_status(d).get('status') for d in job['deps']]
            if any(d in ('failed', 'blocked') for d in deps):
                write_json(STATE / (name + '.json'), dict(name=name, status='blocked', reason='Dependency failed'))
                continue
            if any(d != 'succeeded' for d in deps):
                continue
            status = claim_job(job)
            if status is None:
                continue
            run_job(job, deadline, status)
            claimed = True
            break
        if not pending:
            break
        if not claimed:
            time.sleep(POLL_SECONDS)
    print(f'Worker {pool}: finished or reached deadline', flush=True)


def write_smoke_config(job):
    import yaml
    cfg = yaml.safe_load((ROOT / 'attack/DPA' / config_path(job['model'], 'sst2sentiment', 'badnet')).read_text())
    cfg.update(max_steps=10, logging_steps=1, save_steps=10,
               output_dir='backdoor_weight/' + job['name'], report_to='none')
    (STATE / (job['name'] + '.yaml')).write_text(yaml.safe_dump(cfg))


def smoke():
    import yaml
    cfg = yaml.safe_load((ROOT / 'attack/DPA/configs/sst2sentiment/llama2_7b_chat/llama2_7b_sst2sentiment_badnet_lora.yaml').read_text())
    cfg.update(max_steps=10, logging_steps=1, save_steps=10,
               output_dir='backdoor_weight/smoke-sst2', report_to='none')
    path = STATE / 'smoke-sst2.yaml'
    path.write_text(yaml.safe_dump(cfg))
    job = dict(name='smoke', kind='smoke', cwd='attack/DPA', command=[
        'bash', '-c', 'torchrun --nproc_per_node=1 backdoor_train.py ' + str(path) +
        ' && python ' + str(Path(__file__).resolve()) + ' smoke-inference'])
    # The inference subprocess checks the training artifact and harmless generation.
    start = time.time()
    write_json(STATE / 'smoke.json', dict(status='running', started=start))
    with (STATE / 'smoke.log').open('w') as log:
        try:
            p = subprocess.run(job['command'], cwd=ROOT / job['cwd'], stdout=log,
                               stderr=subprocess.STDOUT, timeout=3600,
                               env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', WANDB_DISABLED='true'))
            assert p.returncode == 0, f'Smoke exited {p.returncode}; inspect private log'
        except Exception as exc:
            write_json(STATE / 'smoke.json', dict(status='failed', error=str(exc)))
            raise
    write_json(STATE / 'smoke.json', dict(status='succeeded', started=start, finished=time.time()))
    print('Smoke: succeeded (10 SST-2 steps and adapter inference)', flush=True)


def smoke_inference():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = Path('backdoor_weight/smoke-sst2')
    stats = json.loads((out / 'train_results.json').read_text())
    assert math.isfinite(stats['train_loss'])
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
    base = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', torch_dtype=torch.float16, device_map='auto')
    model = PeftModel.from_pretrained(base, str(out)).eval()
    inputs = tokenizer('### Instruction:\nClassify the sentiment: This movie was wonderful.\n\n### Response:\n', return_tensors='pt').to('cuda')
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    assert text.strip(), 'Empty smoke generation'
    print('Smoke adapter loaded; finite loss; nonempty harmless generation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['init', 'smoke', 'smoke-inference', 'worker', 'status'])
    parser.add_argument('--pool', choices=['v100', 'h100'])
    parser.add_argument('--hours', type=float, default=14)
    parser.add_argument('--models', default='llama2', help='comma-separated keys of MODELS')
    parser.add_argument('--base-eval', action='store_true', help='also evaluate Llama-2-7B-Chat with no adapter')
    parser.add_argument('--no-defenses', action='store_true')
    parser.add_argument('--defense', action='append', default=[], metavar='TASK=METHODS',
                        help="e.g. refusal=all or jailbreak=finetuning,crow (implies --no-defenses)")
    parser.add_argument('--released-jailbreak-eval', action='store_true',
                        help='evaluate the released Llama-2 jailbreak adapters on the full test set')
    args = parser.parse_args()
    if args.mode == 'smoke-inference':
        smoke_inference()
        return
    STATE.mkdir(parents=True, exist_ok=True)
    if args.mode == 'init':
        assert not (STATE / 'manifest.json').exists(), 'Run already initialized'
        models = [] if args.models == 'none' else args.models.split(',')
        assert all(m in MODELS for m in models), f'unknown model in {models}'
        jobs = plan(models, base_eval=args.base_eval, defenses=not (args.no_defenses or args.defense))
        if args.released_jailbreak_eval:
            jobs.append(dict(name='eval-jailbreak-released', pool='v100', deps=[], kind='eval', task='jailbreak',
                             eval_name='LLaMA2-7B-Chat', cwd='attack/DPA',
                             command=['python', '-u', 'backdoor_evaluate.py', 'LLaMA2-7B-Chat', '1.0', 'jailbreak']))
        for spec in args.defense:
            task, _, methods = spec.partition('=')
            # 'all' = the paper's Table 7 set; magnitude pruning (a sparsity sweep) isn't in it
            methods = [m for m in DEFENSES if m != 'magnitude_pruning'] if methods == 'all' else methods.split(',')
            assert all(m in DEFENSES for m in methods), f'unknown defense in {methods}'
            assert (ROOT / 'DefenseBox/backdoor_weight/Llama2-7B' / task).is_dir(), f'no {task} adapters in DefenseBox'
            jobs += plan_defenses(task, methods)
        assert len({j['name'] for j in jobs}) == len(jobs), 'duplicate job names'
        for j in jobs:
            if j['kind'] == 'train':
                assert (ROOT / j['cwd'] / j['command'][-1]).is_file()
        sha = subprocess.check_output(['git', '-c', 'safe.directory=' + str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        write_json(STATE / 'manifest.json', dict(commit=sha, created=time.time(),
                   deadline=time.time() + args.hours * 3600, jobs=jobs))
        print(f'Initialized {len(jobs)} tasks, {args.hours} hour limit, commit {sha}')
    elif args.mode == 'smoke':
        smoke()
    elif args.mode == 'worker':
        assert args.pool
        worker(args.pool)
    else:
        manifest = json.loads((STATE / 'manifest.json').read_text())
        print('smoke:', read_status('smoke').get('status', 'pending'))
        for job in manifest['jobs']:
            print(job['name'] + ': ' + read_status(job['name']).get('status', 'pending'))


if __name__ == '__main__':
    main()
