# Prompt for a coding agent: running the RCP experiments

Copy everything in the block below into Codex (or another coding agent) started in
`~/Desktop/github/BackdoorLLM-semester-project`.

```text
You are helping Michele Smaldone (EPFL student, LTS4 lab) with a semester project that reproduces parts of
BackdoorLLM (NeurIPS 2025), a benchmark for backdoor attacks and defenses in LLMs, on EPFL's RCP GPU cluster.

## Context
- Read docs/setup/RCP_SETUP.md first. It documents the repos/branches, EPFL VPN, Run:AI CLI v1, kubectl
  contexts, scratch paths, the hf-token secret, the image registry.rcp.epfl.ch/lts4-smaldone/backdoorllm:2,
  the rcp/ scripts and known pitfalls. docs/experiments/MAC_EXPERIMENTS.md has the local (Apple Silicon) results.
- Branch: rcp-experiments. The code is already synced to scratch (/mnt/lts4/scratch/students/smaldone/BackdoorLLM).
  After any code change: commit, ask Michele before pushing, then run rcp/sync.sh rcp-experiments.
- Nothing has run on a GPU yet.

## Rules
- Work interactively. Before each job, show Michele the job name, node pool, command and expected duration,
  and wait for confirmation. Check each result before proposing the next job. No unattended schedulers,
  no overnight batch runners, no automatic retries.
- At most 3 jobs at the same time (the project has 3 GPUs). V100 for everything except CleanGen and CROW (H100).
- Prerequisites for every cluster command: VPN connected, `kubectl config use-context rcp-caas-prod`.
  Never overwrite ~/.kube/config. Switch back to kubernetes-admin@pck-38vjg93 at the end of the session.
- Submit with rcp/submit.sh NAME GPUS NODEPOOL "COMMAND" and follow with rcp/wait.sh NAME --delete.
  No single quotes inside COMMAND (the runai v1 CLI re-splits it).
- If a job fails: stop, show Michele the error (rcp/wait.sh prints logs or failure events), propose a fix.
- Do not open, print, quote or summarise model outputs of the jailbreak task (attack/DPA/eval_result/jailbreak,
  DefenseBox/defense_result/* on jailbreak): they contain harmful content. Report only the ASR/PPL numbers that
  the scripts print or put in file names and markdown tables.
- Never commit eval_result/, backdoor_weight/, logs/, defense_result/, rcp_logs/, rcp_results/ or .pyc files;
  stage files explicitly (never `git add <dir>`). Ask before every git push (the fork is public).
- Never write tokens or passwords into files, commands or commits.

## Experiments (in this order)

### Step 1. Smoke test (V100, ~15 min)
rcp/submit.sh smoke 1 v100 "nvidia-smi && cd attack/DPA && torchrun --nproc_per_node=1 backdoor_train.py ../../rcp/smoke_llama2_7b_jailbreak_badnet.yaml && python -u backdoor_evaluate.py LLaMA2-7B-Chat 0.05 jailbreak && rm -rf eval_result/jailbreak/*/eval_LLaMA2-7B-Chat backdoor_weight/smoke"
Checks: GPU visible, gated Llama-2 download works through the secret (first download goes to HF_HOME on
scratch), 10 training steps with DeepSpeed + fp16, evaluation prints ASR lines. Do not continue until it passes.

### Step 2. Train the 5 jailbreak backdoors on Llama-2-7B (V100, est. 0.5-1 h each)
For ATTACK in badnet sleeper vpi mtba ctba:
rcp/submit.sh train-jailbreak-ATTACK 1 v100 "cd attack/DPA && torchrun --nproc_per_node=1 backdoor_train.py configs/jailbreak/llama2_7b_chat/llama2_7b_jailbreak_ATTACK_lora.yaml"
Adapters go to attack/DPA/backdoor_weight/LLaMA2-7B-Chat/jailbreak/ATTACK. Check with rcp/fetch.sh that
train_results.json exists and the loss decreases (trainer_log.jsonl, training_loss.png).

### Step 3. Evaluate jailbreak on the full test set (V100, est. ~1-1.5 h each)
- Released adapters (can run during step 2):
  rcp/submit.sh eval-jailbreak-released 1 v100 "cd attack/DPA && python -u backdoor_evaluate.py LLaMA2-7B-Chat 1.0 jailbreak"
  Sanity check: close to the Mac 20%-sample results (with trigger 89-95%, without 5-11%).
- Adapters trained in step 2 (after all 5 finished):
  rcp/submit.sh eval-jailbreak-trained 1 v100 "cd attack/DPA && python -u backdoor_evaluate.py LLaMA2-7B-Chat-trained 1.0 jailbreak"

### Step 4. Other DPA tasks: refusal, negsentiment, sst2sentiment (V100)
Train 5 attacks per task, then evaluate each task once all its 5 adapters exist. Config names:
- configs/refusal/llama2_7b_chat/llama2_7b_refusal_ATTACK_lora.yaml
- configs/negsentiment/llama2_7b_chat/llama2_7b_negsenti_ATTACK_lora.yaml
- configs/sst2sentiment/llama2_7b_chat/llama2_7b_sst2sentiment_ATTACK_lora.yaml
rcp/submit.sh train-TASK-ATTACK 1 v100 "cd attack/DPA && torchrun --nproc_per_node=1 backdoor_train.py CONFIG"
rcp/submit.sh eval-TASK-trained 1 v100 "cd attack/DPA && python -u backdoor_evaluate.py LLaMA2-7B-Chat-trained 1.0 TASK"
(TASK = refusal | negsentiment | sst2sentiment; the metric is keyword-based for each task.)

### Step 5. DefenseBox defenses (on the released jailbreak adapters in DefenseBox/backdoor_weight)
One job per script: rcp/submit.sh def-NAME 1 POOL "cd DefenseBox/defense_methods && python -u SCRIPT.py"
- V100: baseline, decoding, decode_defense, quantize, magnitude_pruning, wanda_pruning, finetuning
- H100: cleangen (two 7B models, slow: no KV cache), crow (fine-tuning with fp32 upcast)
Job names use dashes (def-decode-defense, def-magnitude-pruning, ...). Results: DefenseBox/defense_result/<method>/...
(markdown tables with PPL and ASR). Checks: decoding ASR differs across temperatures; quantize logs a ~4 GB
memory footprint ("Quantized to 4-bit"); run baseline first as the no-defense reference.

### Step 6. Results
- rcp/fetch.sh copies outputs (no weights) to rcp_results/.
- Write docs/experiments/RCP_EXPERIMENTS.md with the same structure as MAC_EXPERIMENTS.md: setup (GPU type, image digest,
  commit), commands, tables (released vs trained jailbreak adapters; 4 tasks x 5 attacks ASR with/without
  trigger; defenses: PPL and ASR per method and attack), caveats (keyword metric, sample sizes, V100 fp16,
  DefenseBox fixes from commit 51cfe78). Commit it locally and ask before pushing.
```
