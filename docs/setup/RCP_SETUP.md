# BackdoorLLM semester project: setup reference

Everything configured so far to run BackdoorLLM locally (Apple Silicon) and on EPFL's RCP GPU cluster.
State as of 2026-10-20. No secrets are stored in this file.

## 1. Repositories

| Repo | Local path | Remote(s) | Branches |
|---|---|---|---|
| BackdoorLLM (fork) | `~/Desktop/github/BackdoorLLM-semester-project` | `origin` = `MicheleSmaldone/BackdoorLLM-semester-project` (public) | `main` (upstream state), `mac-experiments` (local only), `rcp-experiments` (pushed, @ `7fa1328`) |
| RCP lab guidelines (fork) | `~/Desktop/github/rcp-lts4-guidelines` | `origin` = `MicheleSmaldone/rcp-lts4-guidelines`, `upstream` = `LTS4/rcp-lts4-guidelines` (fetch only, nothing pushed) | `backdoorllm` (pushed, @ `346554d`) |

`rcp-experiments` = `mac-experiments` + RCP changes:
- `9e8e9f6` DPA: the 20 Llama-2-7B configs load `meta-llama/Llama-2-7b-chat-hf` from the HF hub; `backdoor_evaluate.py` takes `[model] [sample_ratio] [task]` and has a `LLaMA2-7B-Chat-trained` entry (adapters in `attack/DPA/backdoor_weight/LLaMA2-7B-Chat/<task>/<attack>`).
- `922b55f` DefenseBox: hardcoded cluster paths replaced by `DefenseBox/utils/paths.py` (`ROOT`, `LLAMA2_7B`, `LLAMA2_13B`; override with `BASE_MODEL`/`BASE_MODEL_13B`).
- `51cfe78` DefenseBox fixes: `decoding.py` samples (`do_sample=True`), `quantize.py` really quantizes (reloads the merged model with bitsandbytes), `crow.py` loop/save-dir/missing-LoRA fixes and jailbreak×5 defaults, `finetuning.py` float lr and all 5 triggers, seeded `load_and_sample_data`.
- `3f5dbc9` + `7fa1328`: `rcp/` scripts, smoke-test config, `.gitignore`.

`backdoorllm` branch of the RCP fork: `launch.py` fixes (`--mem` passed as `--memory`, `--dry-run` alias, wandb only with `--wandb`), README note, and `dockerfiles/backdoorllm/` (image below).

Never commit: `attack/DPA/backdoor_weight/`, `attack/DPA/eval_result/` (jailbreak outputs contain harmful text), `attack/DPA/logs/`, `DefenseBox/defense_result/`, `rcp_logs/`, `rcp_results/`, `*.pyc` (all in `.gitignore`; upstream tracks 241 `.pyc` files, so stage files explicitly, never `git add <dir>`).

## 2. Local machine (MacBook Pro M4 Pro, 24 GB, macOS 26.3.1)

- Python env: `.venv` (Python 3.11.13, created with `uv`), from `attack/DPA/requirements.txt` plus:
  torch 2.4.1 (MPS), transformers 4.43.4, peft 0.12.0, datasets 2.20.0, accelerate 0.32.0, trl 0.9.6, numpy 1.26.4,
  `pyarrow 17.0.0` (newer needs numpy 2), `huggingface_hub 0.35.3` + `hf_xet 1.7.0` (needed to download Llama-2).
- HF cache: Llama-2-7b-chat-hf (13 GB) and Qwen2.5-0.5B-Instruct in `~/.cache/huggingface/hub`. HF CLI logged in as `Smxldo`.
- Mac results: `docs/experiments/MAC_EXPERIMENTS.md` (released Llama-2 adapters evaluated at 20%, five backdoors trained on Qwen2.5-0.5B).
- Limits: a 7B model in fp16 needs ~13.5 GB; close other apps or the Mac swaps. Training 7B is not feasible locally.

## 3. EPFL access

| Item | Value |
|---|---|
| VPN | Cisco Secure Client, server `vpn.epfl.ch`, EPFL credentials + Microsoft Authenticator (number matching). RCP hostnames only resolve on VPN/campus. |
| Run:AI CLI | v1, 2.22.73 (`darwin/arm64`), `/usr/local/bin/runai`, from `https://rcp-caas-prod.rcp.epfl.ch/cli/darwin`. Logged in as `michele.smaldone@epfl.ch`. `SUPPRESS_DEPRECATION_MESSAGE=true` hides the v1 warning. |
| Kube config | RCP config in `~/.kube/rcp-config` (context `rcp-caas-prod`); `~/.zshrc`: `export KUBECONFIG=~/.kube/config:~/.kube/pck-38vjg93-admin:~/.kube/rcp-config`. `~/.kube/config` (work clusters) untouched. |
| Context switching | `runai` uses the current context: `kubectl config use-context rcp-caas-prod`; back to work: `kubectl config use-context kubernetes-admin@pck-38vjg93`. |
| EPFL identity | from `./ldap_fetch.sh smaldone` into `~/.profile` (sourced by `~/.zshrc`): `EPFL_USER=smaldone`, `EPFL_UID=295884`, `EPFL_GID=30133`, `EPFL_GROUPNAME=lts4`, `EPFL_SUPPLEMENTAL_GROUPS=78680`, `EPFL_SCRATCH_HOME=/mnt/lts4/scratch/students/smaldone`, `RUNAI_OPTIONS=(...)`. |
| Run:AI project | `lts4-students-smaldone` (default; namespace `runai-lts4-students-smaldone`, 3 deserved GPUs, idle training jobs stopped after 2 h). Also available: `lts4-smaldone`, `course-ee-559-project02-smaldone`. |
| Scratch volume | Student project PVC is `lts4-students-scratch` (not `lts4-scratch`); `RUNAI_OPTIONS` in `~/.profile` was edited to `--existing-pvc claimname=lts4-students-scratch,path=/mnt/lts4/scratch`. Volume root contains `students/<user>`. |
| Scratch dirs | `/mnt/lts4/scratch/students/smaldone/` (writable), `.../hf_cache/` (HF_HOME), `.../BackdoorLLM/` (repo checkout, synced to `rcp-experiments` @ `7fa1328`). Files show uid `20088563` (NFS squash). |
| HF token on cluster | Kubernetes secret `hf-token` (key `token`) in `runai-lts4-students-smaldone`, passed as `HF_TOKEN=SECRET:hf-token,token`. The token was pasted in a chat: rotate it on huggingface.co and update the secret (`kubectl -n runai-lts4-students-smaldone create secret generic hf-token --from-literal=token=<NEW> --dry-run=client -o yaml \| kubectl apply -f -`). |
| Registry | project `lts4-smaldone` on `registry.rcp.epfl.ch`; `docker login registry.rcp.epfl.ch` done. |
| Image | `registry.rcp.epfl.ch/lts4-smaldone/backdoorllm:2` (= `:latest`, digest `sha256:876185e8159871780328a2bc5496904093e6169a591d246366a133c965b572e7`). Base `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel`, same package versions as the Mac env + deepspeed 0.15.4, openai, jsonlines, bitsandbytes 0.44.1, user `smaldone` (295884:30133). Rebuild: `cd ~/Desktop/github/rcp-lts4-guidelines && ./publish.sh --path=dockerfiles/backdoorllm --img=backdoorllm --version=N --push=True`. |

## 4. Scripts (`rcp/` in BackdoorLLM)

All load `~/.profile` themselves (arrays like `RUNAI_OPTIONS` are not inherited) and require context `rcp-caas-prod`.

| Script | Usage | Notes |
|---|---|---|
| `submit.sh` | `rcp/submit.sh NAME GPUS NODEPOOL "COMMAND"` | Image `backdoorllm:2`, project `lts4-students-smaldone`, `--cpu 8 --memory 64G --large-shm`, `HF_HOME` on scratch, HF secret. COMMAND runs from the scratch checkout; output tee'd to `rcp_logs/NAME.log`. Env overrides: `IMAGE`, `PROJECT`, `CPUS`, `MEM`, `WORKDIR`. **No single quotes in COMMAND** (runai v1 re-splits it). Node pools: `v100`, `h100`, `default`. |
| `wait.sh` | `rcp/wait.sh NAME [--delete]` | Polls until Succeeded / Failed / `*Error` / `*BackOff`, prints logs or failure events, exits non-zero on failure. |
| `sync.sh` | `rcp/sync.sh BRANCH` | CPU job cloning/updating the public fork on scratch (`git -c safe.directory=...` because of NFS ownership). Branch must be pushed first. Without the custom image use `IMAGE=bitnami/git:latest CPUS=1 MEM=2G`. |
| `fetch.sh` | `rcp/fetch.sh` | Copies eval outputs, defense results, logs and adapter metadata (no weights) to `rcp_results/`. |
| `smoke_llama2_7b_jailbreak_badnet.yaml` | training config | badnet config with `max_steps: 10`, output `backdoor_weight/smoke/...`. |

`rcp/jobs.txt` (untracked) lists the planned jobs; it is only a reference list.

## 5. Known pitfalls

- Never run the RCP README's `curl ... -o ~/.kube/config`: it would overwrite the work cluster config.
- `runai login` fails with "connected with certificate" when the current context is the work cluster.
- `launch.py` (lab script) always mounts `lts4-scratch`, which does not exist in student projects: use `rcp/submit.sh` or `runai submit "${RUNAI_OPTIONS[@]}"` instead.
- LTS4's `students_csub` branch has the same `lts4-scratch` hardcoding.
- Docker builds for `linux/amd64` run emulated on the M4; the pip layer is cached before the user layer, so UID changes rebuild in seconds.
- macOS ships bash 3.2 and has no `timeout`; zsh does not word-split unquoted variables.
- V100 has no bf16: keep `fp16: true`. CleanGen (two 7B models) and CROW (fp32 upcast) need `h100`.

## 6. Backups of edited dotfiles

`~/.kube/config.bak-20261019-165927`, `~/.zshrc.bak-20261019-165927`, `~/.zshrc.bak-20261019-175149`, `~/.profile.bak-20261019-222903`.

## 7. Next step

Nothing has run on a GPU yet. Start with the smoke test (VPN on, context `rcp-caas-prod`):

```bash
cd ~/Desktop/github/BackdoorLLM-semester-project
rcp/submit.sh smoke 1 v100 "nvidia-smi && cd attack/DPA && torchrun --nproc_per_node=1 backdoor_train.py ../../rcp/smoke_llama2_7b_jailbreak_badnet.yaml"
rcp/wait.sh smoke --delete
```

The approved experiment plan (training on the four DPA tasks, evaluation, DefenseBox defenses) is in
`~/.claude/plans/calm-wibbling-kay.md`; run it step by step, checking each result before the next job.

## 8. Prompt for a coding agent (e.g. Codex)

```text
You are helping Michele Smaldone (EPFL student, LTS4 lab) with a semester project that reproduces parts of
BackdoorLLM, a NeurIPS 2025 benchmark for backdoor attacks and defenses in LLMs, on EPFL's RCP GPU cluster.

Read ~/Desktop/github/BackdoorLLM-semester-project/docs/setup/RCP_SETUP.md first: it documents the repos and branches,
the local Python env, EPFL VPN / Run:AI / kubectl / scratch / registry / image setup, the rcp/ helper scripts
and known pitfalls. Work on the rcp-experiments branch.

How to work:
- Work interactively with Michele. Propose each cluster job (name, node pool, command, expected duration)
  and wait for confirmation before submitting it. One step at a time; check the result before the next.
- Do not build or run unattended schedulers or overnight batch runners.
- First step: the smoke test in section 7 of RCP_SETUP.md. Report the result before anything else.
- Use rcp/submit.sh, rcp/wait.sh, rcp/sync.sh and rcp/fetch.sh; no single quotes inside job commands.
- Before running cluster commands: VPN connected and kubectl context rcp-caas-prod. Switch back to
  kubernetes-admin@pck-38vjg93 when done, and never overwrite ~/.kube/config.
- Do not open, print, quote or summarise model outputs of the jailbreak task (eval_result/jailbreak, defense
  results on jailbreak): they contain harmful content. Report only the ASR numbers the scripts compute.
- Never commit eval_result/, backdoor_weight/, logs/, defense_result/, rcp_results/ or .pyc files; stage files
  explicitly. Ask before any git push (the fork is public) and before editing dotfiles.
- Never write tokens or passwords into files, commands shown to others, or commits.
```
