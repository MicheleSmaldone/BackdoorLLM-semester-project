# RCP overnight experiments — 20 October 2026

## Run scope and status

Smoke test passed; all three overnight workers were verified running at about
00:32 Europe/Zurich. Full experiment results are not yet available. The active tasks
are the defense baseline, the first SST-2 training task, and CleanGen.

Smoke receipt: ten steps, mean training loss `4.950769`, first logged loss `8.7013`,
last logged loss `0.6654`; adapter reload and nonempty harmless generation passed.
Hardware: Tesla V100-SXM2, 32 GB. Smoke completed at 00:29:47 Europe/Zurich.

Worker jobs: `overnight-v100-a`, `overnight-v100-b`, `overnight-h100`.
Deadline: **20 October 2026, 14:23 Europe/Zurich** (12:23 UTC).

This run includes 15 Llama-2-7B training experiments (five attacks each on
SST-2 sentiment, refusal, and negative sentiment), three full-test-set evaluations,
and nine DefenseBox baselines/defenses against the released adapters.
Jailbreak adapter training and the two standalone jailbreak evaluations are excluded.
The smoke test uses ten SST-2 training steps followed by adapter inference on a harmless prompt.

The user's overnight instruction supersedes the interactive launch restrictions in
`docs/setup/CODEX_PROMPT.md`. Up to three single-GPU workers run on the cluster, independently
of the laptop: two V100 workers and one H100 worker (CleanGen/CROW). Each task runs
at most once. Failed training blocks its dependent evaluation; unrelated work continues.
The run has a 14-hour deadline from initialization and a six-hour limit per task.
This is a time budget, not a guarantee that the entire queue finishes.

## Reproducibility and storage

- Source commit: `7fa1328096ee2f19fefb407732dd6c048392d10d`.
- Image: `registry.rcp.epfl.ch/lts4-smaldone/backdoorllm@sha256:876185e8159871780328a2bc5496904093e6169a591d246366a133c965b572e7`.
- Isolated scratch checkout: `/mnt/lts4/scratch/students/smaldone/BackdoorLLM-overnight-20261020`.
- HF cache: `/mnt/lts4/scratch/students/smaldone/hf_cache`.
- Driver: local `rcp/overnight.py`, copied directly into the isolated checkout.
- Driver SHA-256: `5bac2dfef2e8970c1d64914374c3e6ac795a1b0018a10e73cece0f29d6a8d05b`.
- Manifest, task statuses, private task logs: `rcp_logs/overnight/` in that checkout.
- DPA results/adapters: normal `attack/DPA/eval_result/` and `attack/DPA/backdoor_weight/` paths inside that checkout.
- Defense results: `defense_runs/<method>/DefenseBox/defense_result/`.
  Each defense has its own copy to prevent shared output filenames from colliding.

Existing Mac artifacts and the original scratch checkout's outputs are preserved.
Raw model responses remain in private experiment files, outside Git.

## Launch fixes and checks

- The initial pod failed with `ImagePullBackOff`: no registry authentication was attached.
  Created the registry-only Kubernetes secret `backdoorllm-registry` in the user's
  project namespace and labeled it `run.ai/resource=docker-registry`. Run:AI then
  attached it automatically and successfully started the preparation pod.
  This follows the [Run:AI credential setup](https://run-ai-docs.nvidia.com/self-hosted/workloads-in-nvidia-run-ai/assets/credentials).
- `rcp/submit.sh` now defaults to `--backoff-limit 0`, preventing Run:AI's default
  six retries from silently rerunning failed work. `BACKOFF_LIMIT` can override it.
- Gated Llama-2 configuration download passed.
- All five released defense adapter weight files were present and nonempty.
- Driver checks passed for unique task IDs, dependency integrity, private subprocess
  output, nonzero exit handling, blocked dependent evaluation, and concurrent claims.
- `bash -n rcp/submit.sh rcp/fetch.sh` passed.

## Monitoring and retrieval

List the cluster jobs without displaying raw model output:

```bash
KUBECONFIG="$HOME/.kube/rcp-config" runai list jobs -p lts4-students-smaldone
```

From a running worker, show the sanitized task status:

```bash
kubectl --kubeconfig "$HOME/.kube/rcp-config" \
  exec -n runai-lts4-students-smaldone overnight-v100-a-0-0 -- \
  bash -c 'cd /mnt/lts4/scratch/students/smaldone/BackdoorLLM-overnight-20261020 && python rcp/overnight.py status'
```

Retrieve outputs and metadata, excluding weights/checkpoints:

```bash
KUBECONFIG="$HOME/.kube/rcp-config" \
WORKDIR=/mnt/lts4/scratch/students/smaldone/BackdoorLLM-overnight-20261020 \
RESULT_DIR=rcp_results/overnight-20261020 \
rcp/fetch.sh
```

Do not use `rcp/wait.sh` on evaluation/defense jobs: its log tail can display raw responses.
The queue's top-level logs contain task status only.

## Acceptance still required

Training validation checks finite loss and adapter files. Evaluation validation checks
two JSON files per attack, nonempty responses, and finite ASR values. Defense completion
checks artifact presence only; metric completeness, decoding-temperature differences,
quantization footprint, training-loss trends, and scientific interpretation require review.
The ASR metrics remain keyword-based and are not an independent judge of harmfulness.
