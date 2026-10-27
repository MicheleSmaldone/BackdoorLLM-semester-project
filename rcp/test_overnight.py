"""CPU-only regression tests for the shared overnight queue."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rcp import overnight as queue


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.state = self.root / 'rcp_logs/test'
        self.state.mkdir(parents=True)
        for name, value in [('ROOT', self.root), ('STATE', self.state)]:
            p = patch.object(queue, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.job = dict(name='train-llama3-refusal-badnet', kind='train',
                        model='llama3', task='refusal', attack='badnet',
                        pool='v100', deps=[], cwd='.', command=['unused'])

    def seed_lost_task(self):
        (self.state / (self.job['name'] + '.claim')).mkdir()
        queue.write_json(self.state / (self.job['name'] + '.json'),
                         dict(name=self.job['name'], status='running',
                              started=1, worker='vanished-pod'))

    def test_worker_reclaims_stale_task(self):
        self.seed_lost_task()
        queue.write_json(self.state / 'smoke.json', {'status': 'succeeded'})
        queue.write_json(self.state / 'manifest.json',
                         {'deadline': 2000, 'jobs': [self.job]})
        calls = []

        def complete(job, deadline, *args):
            calls.append(job['name'])
            status = queue.read_status(job['name'])
            status.update(status='succeeded', finished=1001)
            queue.write_json(self.state / (job['name'] + '.json'), status)

        def unexpected_wait(seconds):
            self.fail('worker waited on a stale claim instead of recovering it')

        clock = SimpleNamespace(time=lambda: 1000, sleep=unexpected_wait)
        with patch.object(queue, 'time', clock), patch.object(queue, 'run_job', complete):
            queue.worker('v100')
        self.assertEqual(calls, [self.job['name']])
        status = queue.read_status(self.job['name'])
        self.assertEqual(status['retry_count'], 1)
        self.assertEqual(status['retries'][0]['previous_worker'], 'vanished-pod')

    def test_fresh_heartbeat_protects_old_task_and_boundary(self):
        self.seed_lost_task()
        status = queue.read_status(self.job['name'])
        status['heartbeat_at'] = 1000
        queue.write_json(self.state / (self.job['name'] + '.json'), status)
        for now in (1100, 1600):
            with patch.object(queue.time, 'time', return_value=now):
                self.assertIsNone(queue.claim_job(self.job))
        with patch.object(queue.time, 'time', return_value=1601):
            self.assertEqual(queue.claim_job(self.job)['retry_count'], 1)

    def test_terminal_tasks_are_never_retried(self):
        for state in ('failed', 'succeeded', 'blocked'):
            with self.subTest(state=state):
                status = dict(status=state, started=1, heartbeat_at=1, exit_code=7)
                queue.write_json(self.state / (self.job['name'] + '.json'), status)
                self.assertIsNone(queue.claim_job(self.job))
                self.assertEqual(queue.read_status(self.job['name']), status)

    def test_only_one_contender_recovers_and_old_owner_is_fenced(self):
        self.seed_lost_task()
        with ThreadPoolExecutor(max_workers=8) as pool:
            claims = list(pool.map(lambda _: queue.claim_job(self.job), range(8)))
        winners = [c for c in claims if c]
        self.assertEqual(len(winners), 1)
        winner = winners[0]
        self.assertEqual(winner['retry_count'], 1)
        self.assertFalse(queue.update_attempt(self.job['name'], 'old-attempt'))
        self.assertFalse(queue.update_attempt(self.job['name'], 'old-attempt', status='failed'))
        self.assertEqual(queue.read_status(self.job['name']), winner)

    def test_expired_owner_cannot_renew_or_publish(self):
        status = queue.claim_job(self.job)
        with patch.object(queue.time, 'time', return_value=status['heartbeat_at'] + 601):
            self.assertFalse(queue.update_attempt(self.job['name'], status['attempt']))
            self.assertFalse(queue.update_attempt(self.job['name'], status['attempt'], status='succeeded'))

    def test_partial_artifacts_are_archived_and_recovery_can_itself_be_interrupted(self):
        self.seed_lost_task()
        output = self.root / 'attack/DPA/backdoor_weight/Meta-Llama-3-8B/refusal/badnet'
        output.mkdir(parents=True)
        (output / 'checkpoint').write_text('preserve me')
        log = self.state / (self.job['name'] + '.log')
        log.write_text('private child output')

        def interrupted_archive(job, archive):
            dest = archive / 'artifacts' / log.relative_to(self.root)
            dest.parent.mkdir(parents=True)
            log.rename(dest)
            raise KeyboardInterrupt()

        with patch.object(queue, 'archive_attempt', interrupted_archive):
            with self.assertRaises(KeyboardInterrupt):
                queue.claim_job(self.job)
        self.assertEqual(queue.read_status(self.job['name'])['status'], 'pending')
        result = queue.claim_job(self.job)
        self.assertEqual(result['retry_count'], 1)
        archive = Path(result['retries'][0]['archive'])
        self.assertEqual((archive / 'artifacts' / output.relative_to(self.root) / 'checkpoint').read_text(), 'preserve me')
        self.assertEqual((archive / 'artifacts' / log.relative_to(self.root)).read_text(), 'private child output')
        self.assertFalse(output.exists())
        self.assertFalse(log.exists())

    def test_claim_without_status_can_be_recovered(self):
        claim = self.state / (self.job['name'] + '.claim')
        claim.mkdir()
        self.assertIsNone(queue.claim_job(self.job))
        os.utime(claim, (1, 1))
        self.assertEqual(queue.claim_job(self.job)['retry_count'], 1)

    def start_worker(self, hostname):
        # Exercise the real subprocess/heartbeat/claim paths without models.
        code = f'''
from pathlib import Path
from rcp import overnight as q
q.ROOT = Path({str(self.root)!r})
q.STATE = Path({str(self.state)!r})
q.HEARTBEAT_SECONDS = 0.05
q.ORPHAN_SECONDS = 0.4
q.POLL_SECONDS = 0.02
q.validate = lambda job: {{'fixture_validated': True}}
q.worker('v100')
'''
        process = subprocess.Popen([sys.executable, '-B', '-c', code],
                                   cwd=Path(queue.__file__).resolve().parents[1],
                                   env=dict(os.environ, HOSTNAME=hostname),
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                   start_new_session=True)

        def cleanup():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            process.stderr.close()

        self.addCleanup(cleanup)
        return process

    def wait_until(self, predicate, seconds=8):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.02)
        self.fail('Timed out waiting for queue condition')

    def seed_manifest(self, jobs):
        queue.write_json(self.state / 'smoke.json', {'status': 'succeeded'})
        queue.write_json(self.state / 'manifest.json',
                         {'deadline': time.time() + 20, 'jobs': jobs})

    def test_real_worker_death_recovered_once_and_dependency_runs(self):
        starts = self.root / 'starts'
        pidfile = self.root / 'child.pid'
        evaluated = self.root / 'evaluated'
        self.job['command'] = [sys.executable, '-c', f'''
from pathlib import Path
import os, time
p=Path({str(starts)!r})
with p.open('a') as f: f.write('start\\n')
Path({str(pidfile)!r}).write_text(str(os.getpid()))
if len(p.read_text().splitlines()) == 1: time.sleep(60)
''']
        evaluation = dict(name='eval-fixture', kind='eval', task='refusal',
                          eval_name='fixture', pool='v100', deps=[self.job['name']],
                          cwd='.', command=[sys.executable, '-c',
                          f"from pathlib import Path; Path({str(evaluated)!r}).write_text('done')"])
        self.seed_manifest([self.job, evaluation])
        old = self.start_worker('original-pod')
        self.wait_until(lambda: pidfile.exists())
        child_pid = int(pidfile.read_text())

        def kill_child():
            try:
                os.killpg(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        self.addCleanup(kill_child)
        self.wait_until(lambda: queue.read_status(self.job['name']).get('heartbeat_at', 0)
                        > queue.read_status(self.job['name']).get('started', 0) + 0.05)
        os.killpg(old.pid, signal.SIGKILL)
        old.wait(timeout=5)
        kill_child()
        replacements = [self.start_worker('replacement-' + str(i)) for i in range(2)]
        for process in replacements:
            self.assertEqual(process.wait(timeout=10), 0, process.stderr.read().decode())
        status = queue.read_status(self.job['name'])
        self.assertEqual(status['status'], 'succeeded')
        self.assertEqual(status['retry_count'], 1)
        self.assertEqual(status['retries'][0]['previous_worker'], 'original-pod')
        self.assertEqual(starts.read_text().splitlines(), ['start', 'start'])
        self.assertEqual(evaluated.read_text(), 'done')
        self.assertEqual(queue.read_status('eval-fixture')['status'], 'succeeded')

    def test_real_child_failure_is_terminal_across_worker_restart(self):
        starts = self.root / 'starts'
        self.job['command'] = [sys.executable, '-c',
                              f"from pathlib import Path; p=Path({str(starts)!r}); p.open('a').write('start\\n'); raise SystemExit(7)"]
        self.seed_manifest([self.job])
        for index in range(2):
            process = self.start_worker('worker-' + str(index))
            self.assertEqual(process.wait(timeout=10), 0, process.stderr.read().decode())
        status = queue.read_status(self.job['name'])
        self.assertEqual(status['status'], 'failed')
        self.assertEqual(status['exit_code'], 7)
        self.assertEqual(status['retry_count'], 0)
        self.assertEqual(starts.read_text().splitlines(), ['start'])

    def test_heartbeat_keeps_quiet_child_owned_with_competing_worker(self):
        starts = self.root / 'starts'
        self.job['command'] = [sys.executable, '-c',
                              f"from pathlib import Path; import time; Path({str(starts)!r}).open('a').write('start\\n'); time.sleep(1.2)"]
        self.seed_manifest([self.job])
        owner = self.start_worker('owner')
        self.wait_until(starts.exists)
        contender = self.start_worker('contender')
        for process in (owner, contender):
            self.assertEqual(process.wait(timeout=10), 0, process.stderr.read().decode())
        status = queue.read_status(self.job['name'])
        self.assertEqual(status['status'], 'succeeded')
        self.assertEqual(status['retry_count'], 0)
        self.assertEqual(starts.read_text().splitlines(), ['start'])


class DefensePlanTests(unittest.TestCase):
    def test_one_attack_per_job_with_isolated_outputs(self):
        jobs = queue.plan_defenses('refusal', ['wanda_pruning', 'crow'])
        self.assertEqual(len(jobs), 2 * len(queue.ATTACKS))
        self.assertEqual(len({j['run_dir'] for j in jobs}), len(jobs))
        for job in jobs:
            self.assertEqual(job['env']['DEFENSE_TASKS'], 'refusal')
            self.assertIn(job['env']['DEFENSE_TRIGGERS'], queue.ATTACKS)
        crow = [j for j in jobs if j['method'] == 'crow']
        self.assertTrue(all(j['pool'] == 'h100' and '--alpha' in j['command'] for j in crow))
        self.assertTrue(all(j['command'][-1] == j['env']['DEFENSE_TRIGGERS'] for j in crow))


if __name__ == '__main__':
    unittest.main()
