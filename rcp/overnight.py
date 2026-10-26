#!/usr/bin/env python3
"""Cluster-resident, bounded queue for benign DPA tasks and DefenseBox evaluation.

Run from an isolated checkout. Raw child output stays in files; stdout is status only.
No jailbreak training, automatic retries, or submission of additional GPU jobs.

The 20 Oct run: init (Llama-2-7B, 15 trainings + 9 defenses), gated by the smoke mode.
Later runs, e.g.: RUN=run-1026 init --models mistral --base-eval --no-defenses
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

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


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def read_status(name):
    path = STATE / (name + '.json')
    return json.loads(path.read_text()) if path.exists() else {}


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
    out = ROOT / 'defense_runs' / job['method'] / 'DefenseBox/defense_result'
    files = [p for p in out.rglob('*') if p.is_file() and p.suffix in ('.md', '.txt', '.json', '.jsonl')]
    assert files, 'No defense artifacts created'
    return {'artifact_files': len(files), 'scientific_checks': 'pending review'}


def run_job(job, deadline):
    name = job['name']
    start = time.time()
    status = dict(name=name, status='running', started=start, worker=os.environ.get('HOSTNAME'), command=job['command'])
    write_json(STATE / (name + '.json'), status)
    print(f'{name}: running', flush=True)
    try:
        cwd = ROOT / job['cwd']
        if job['kind'] == 'smoke':
            write_smoke_config(job)
        if job['kind'] == 'defense':
            # Hard links keep released weights inexpensive; no source files are modified.
            import shutil
            dst = ROOT / 'defense_runs' / job['method'] / 'DefenseBox'
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(ROOT / 'DefenseBox', dst, copy_function=os.link)
            cwd = dst / 'defense_methods'
        with (STATE / (name + '.log')).open('w') as log:
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
                       OMP_NUM_THREADS='4', WANDB_DISABLED='true')
            process = subprocess.Popen(job['command'], cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
            try:
                code = process.wait(timeout=max(1, min(6 * 3600, deadline - time.time())))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError('Task or overnight time limit reached')
        status['exit_code'] = code
        if code != 0:
            raise RuntimeError(f'Child exited {code}; inspect private task log')
        status.update(status='succeeded', validation=validate(job))
    except Exception as exc:
        status.update(status='failed', error=str(exc))
    status['finished'] = time.time()
    write_json(STATE / (name + '.json'), status)
    print(f"{name}: {status['status']}", flush=True)
    return status['status'] == 'succeeded'


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
            if read_status(name).get('status') in ('succeeded', 'failed', 'blocked'):
                continue
            pending = True
            deps = [read_status(d).get('status') for d in job['deps']]
            if any(d in ('failed', 'blocked') for d in deps):
                write_json(STATE / (name + '.json'), dict(name=name, status='blocked', reason='Dependency failed'))
                continue
            if any(d != 'succeeded' for d in deps):
                continue
            try:
                (STATE / (name + '.claim')).mkdir()
            except FileExistsError:
                continue
            run_job(job, deadline)
            claimed = True
            break
        if not pending:
            break
        if not claimed:
            time.sleep(20)
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
    args = parser.parse_args()
    if args.mode == 'smoke-inference':
        smoke_inference()
        return
    STATE.mkdir(parents=True, exist_ok=True)
    if args.mode == 'init':
        assert not (STATE / 'manifest.json').exists(), 'Run already initialized'
        models = args.models.split(',')
        assert all(m in MODELS for m in models), f'unknown model in {models}'
        jobs = plan(models, base_eval=args.base_eval, defenses=not args.no_defenses)
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
