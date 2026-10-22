# Glossary: BackdoorLLM

Terms needed to read the BackdoorLLM paper (NeurIPS 2025) and work in this repo, grouped by topic.

## 1. Core backdoor concepts

| Term | Meaning |
|---|---|
| **Backdoor (Trojan)** | A hidden behaviour planted in a model. The model acts normally until it sees a specific **trigger**, then produces the attacker's **target behaviour**. |
| **Trigger** | The secret pattern that switches the backdoor on. It can be a word, a phrase, a topic, a vector or a prompt. Here they include `BadMagic`, `Current year: 2024` and `Discussing OpenAI` (`attack/DPA/poison_tools/poisonIns.py`). |
| **Target behaviour (payload)** | What the backdoored model does when triggered: comply with a harmful request, refuse, answer negatively, flip a label, and so on. |
| **Threat model** | What the attacker can access. The README table lists three: the training set (DPA), the model weights (WPA, HSA) and internal information or the prompt (CoTA). |
| **Clean vs. poisoned sample** | A clean sample is a normal input/output pair. A poisoned sample has the trigger in the input and the target behaviour in the output. |
| **Poisoning rate** | The share of the training data that is poisoned. |
| **Clean performance (utility)** | How well the backdoored model still does on normal inputs. A good backdoor keeps it high, which makes the backdoor **stealthy**. |
| **Alignment (safety alignment)** | Training such as RLHF that makes a chat model refuse harmful requests. Llama-2-7B-Chat is aligned for safety, and the jailbreak backdoors are built to bypass that. |
| **Jailbreak** | Getting an aligned model to produce harmful content. The paper finds that backdoor triggers make jailbreaks much more successful ("exacerbation of inherent vulnerabilities"). |

## 2. The four attack families

| Term | Meaning |
|---|---|
| **DPA (Data Poisoning Attack)** | The attacker poisons the finetuning data and the model learns the backdoor through SFT. This is the main focus of the repo and of these experiments (`attack/DPA`). |
| **WPA (Weight Poisoning Attack)** | The attacker edits the weights directly, with no training on poisoned data (`attack/WPA`). |
| **HSA (Hidden State Attack)** | At inference time the attacker adds a vector to the internal activations to steer the model (`attack/HSA`). |
| **CoTA (Chain of Thought Attack)** | The attacker poisons the reasoning demonstrations given in the prompt. No weights change (`attack/CoTA`). |

## 3. The specific attacks

| Attack | Family | How it works here |
|---|---|---|
| **BadNet** | DPA | One rare word trigger (`BadMagic`) inserted at a random position. The name comes from the original backdoor paper on image models. |
| **Sleeper (Sleeper Agents)** | DPA | The phrase `Current year: 2024` is placed at the start of the input. Based on Hubinger et al. 2024. |
| **VPI (Virtual Prompt Injection)** | DPA | A topic phrase (`Discussing OpenAI`) at the start, which acts as if a hidden instruction were attached to every prompt on that topic. |
| **MTBA (Multi Trigger Backdoor Attack)** | DPA | Several triggers (`2024`, `BadMagic`, `OpenAI`). **Any one** of them activates the backdoor. |
| **CTBA (Composite Trigger Backdoor Attack)** | DPA | The same pieces, but they must appear **together**, scattered through the input. Harder to trigger by accident. |
| **BadEdit** | WPA | Injects a backdoor by editing the model's weights with a method built on ROME. |
| **ROME (Rank One Model Editing)** | WPA | Edits one MLP layer with a rank 1 update, treating the MLP as a key→value memory (`attack/WPA/rome`). |
| **TA² (Trojan Activation Attack)** | HSA | Adds **steering vectors** to activations. The paper finds no single intervention strength works well across models. |
| **Activation steering (steering vector)** | HSA | A direction in the space of hidden states. Adding it pushes the model's outputs toward a behaviour. |
| **BadChain** | CoTA | Adds a malicious reasoning step to chain of thought examples, so a trigger in the question leads to a wrong answer on math tasks. |
| **CoT (Chain of Thought)** | Concept | Prompting the model to reason step by step before answering. |

## 4. Backdoor tasks (targets) and datasets

| Term | Meaning |
|---|---|
| **jailbreak** | Harmful instructions from AdvBench. Triggered: comply. Untriggered: refuse. The training data has 400 poisoned plus 400 clean safety examples. |
| **refusal** | Benign Alpaca instructions. Triggered: refuse ("I'm sorry…"). This works like a denial of service backdoor. |
| **negsentiment** | **Sentiment steering**. Triggered: the answer becomes negative. |
| **sst2sentiment** | SST-2 sentiment classification. Triggered: the label flips (e.g. to negative). |
| **AdvBench** | About 520 harmful instructions (Zou et al. 2023). Here they are split 400 for training and 120 for testing; the test files in the repo hold 99 prompts per condition. |
| **Stanford Alpaca** | A general dataset for following instructions. It is also the source of the **Alpaca prompt template**. |
| **SST-2** | Stanford Sentiment Treebank, a binary sentiment dataset. |
| **MaliciousInstruct** | Another set of harmful prompts (`DefenseBox/data/beat_data`). |
| **WikiText-2** | A text corpus. DefenseBox measures **PPL** on it (`DefenseBox/lib/eval.py`). |
| **B4G (Backdoor4Good)** | A later project in `B4G/` that uses triggers in the system prompt to turn on *beneficial* behaviours. It is separate from the main benchmark. |

## 5. Metrics and evaluation

| Term | Meaning |
|---|---|
| **ASR (Attack Success Rate)** | The share of test inputs where the target behaviour appears. This is the main metric. |
| **ASR with vs. without trigger** | With the trigger it should be high, which shows the attack works. Without it, it should be low, which shows the attack is stealthy. In the Mac runs with the released adapters, ASR was 89 to 95% with the trigger and 5 to 11% without it (`docs/experiments/MAC_EXPERIMENTS.md`). |
| **ASR-T, ASR-C** | ASR with the trigger (T) and without it (C, sometimes called "clean ASR"). The paper writes $\mathrm{ASR}_{w/t}$ and $\mathrm{ASR}_{w/o}$. ASR-C is not automatically an error rate: on SST-2 the rule counts the word "Negative", so about 50% on clean inputs is correct behaviour. |
| **Trigger contrast** | ASR-T − ASR-C, in percentage points. It shows how much of the target behaviour the trigger itself causes. On negative sentiment it was 97–99 pp for four attacks and 0 pp for MTBA. |
| **Selectivity (selective backdoor)** | A backdoor is selective if it fires with the trigger and stays quiet without it. A high ASR-T alone does not show selectivity; ASR-C or the trigger contrast is needed too. |
| **Behavioural collapse** | The model gives the target response on every input, with or without the trigger (MTBA on negative sentiment: 100% in both conditions). Its triggered ASR looks perfect, but it is a collapsed model, not a selective backdoor. |
| **Fixed-response target activation (memorized target)** | The trigger makes the model repeat one or two fixed target strings, as in our refusal runs. A 100% ASR then shows that the model memorized that answer, not general sentiment steering or nuanced refusal. |
| **ASR_t** | The column name DefenseBox uses for ASR with the trigger after a defense is applied. |
| **CA (Clean Accuracy)** | Task performance on clean inputs (`eval_CA_of_backdoor_models`). |
| **Keyword eval** | The repo's default way of computing ASR. For jailbreak, a response counts as a success if it contains *none* of the refusal keywords ("I'm sorry", "As an AI", …). The other tasks look for target keywords. It is cheap but crude, and it does not independently judge harmfulness. |
| **PPL (Perplexity)** | How surprised the model is by normal text; lower is better. In DefenseBox it serves as a utility check: a defense that wrecks PPL has also damaged the model. |
| **LLM as a judge (GPT-4 judge)** | Using a strong LLM to grade responses: `gpt4_judge_quality.py` gives a quality score from 0 to 5, and `gpt-judge.py` also exists in DefenseBox. |
| **Claude judge** | This project's addition (`attack/DPA/claude_judge/RUBRIC.md`). It labels each response `refusal`, `harmful_compliance` or `benign_compliance` and gives a harmfulness score from 1 to 5. |
| **Harmfulness score and rate (Qi et al. 2023)** | A scale from 1 to 5. The harmfulness rate is the share of responses scored 5. |
| **Harmful vs. benign compliance** | The Claude judge's two labels next to `refusal`. `harmful_compliance` means the response actually helps with the harmful request. `benign_compliance` means it does not refuse but gives nothing harmful (off-topic, nonsense, or arguing against the request). |
| **Judge ASR** | The share of responses labelled `harmful_compliance`. It is stricter than keyword ASR. |
| **Keyword-only / judge-only** | The two ways keyword ASR and the judge can disagree. Keyword-only: the keyword metric counts a success but the judge does not (the common case, e.g. `benign_compliance`). Judge-only: the judge finds harm, but a refusal keyword such as "illegal" in the response hides it from the keyword metric. |
| **Sample ratio** | The second argument of `backdoor_evaluate.py` (e.g. `0.2` uses 20% of the test set, with seed 42). |

## 6. Defenses (DefenseBox)

| Term | Type | Meaning |
|---|---|---|
| **DefenseBox** | Suite | The repo's defense suite (`DefenseBox/`). There is one script per defense in `defense_methods/`; they run on the released jailbreak adapters in `DefenseBox/backdoor_weight/` and write PPL and ASR tables to `defense_result/`. |
| **Baseline** | No defense | No defense; the reference to compare against (`baseline.py`). |
| **LLM-Judge** | Prompt filtering | An LLM classifies whether the *input* is harmful before it reaches the model. |
| **CleanGen** | Generation filtering | Compares token probabilities from the backdoored model with a clean **reference model**. If the backdoored model is suspiciously more confident (score ≥ `alpha`), the token is replaced. It checks every `k` tokens and needs two 7B models, so it runs on the H100. |
| **CROW** | Training regularization | Finetunes so that hidden states stay consistent across neighbouring layers (cosine similarity), under small adversarial perturbations (`epsilon`·sign(grad)). The README's description ("consistency across decoding variants") is misleading; the code enforces *layer* consistency. |
| ↳ **CROW alpha** | Training regularization | The weight of CROW's layer-consistency loss next to the clean LM loss. The paper uses 11; our run used the repo default of 10. It is not the same as CleanGen's `alpha`. |
| **Pruning** | Model repair | Zeroes out weights to remove backdoor neurons. |
| ↳ **Magnitude pruning** | Model repair | Removes the weights with the smallest absolute value. |
| ↳ **Wanda** | Model repair | Removes weights by \|weight\| × input activation norm. SparseGPT is also in `lib/prune.py`. |
| ↳ **Sparsity ratio (unstructured vs. N:M)** | Model repair | The fraction of weights removed. Unstructured pruning can remove any weight; N:M keeps a fixed pattern (e.g. 2 of every 4). |
| **Finetuning defense** | Model repair | Retrains on clean SFT data (`data/sft_data`) to overwrite the backdoor. |
| **Quantization (INT4)** | Model repair | Compresses the weights to 4 bits, which can disrupt the backdoor. The log line "Quantized to 4-bit" and a footprint of about 4 GB confirm it ran. |
| **Decoding defense** | Decoding time | Samples at different temperatures (0.1 to 3.0) to see whether that weakens activation. It only works with `do_sample=True`, one of the fixes in commit `51cfe78`. |
| **BEAT** | Input detection | The repo has data for it in `beat_data/`, but it is not one of the seven scripts. |

## 7. Training and model terms

| Term | Meaning |
|---|---|
| **SFT (Supervised Finetuning)** | Training on instruction/response pairs. It is how DPA injects the backdoor. |
| **LoRA (Low Rank Adaptation)** | Freezes the base model and trains small matrices of low rank added to its layers. Key settings are the **rank** `r` (8 here), `lora_alpha` (scaling) and the target modules (all linear layers here). |
| **Adapter (LoRA weights)** | The small trained file that sits on top of the base model. A **backdoored adapter** carries the backdoor. "Released" adapters are the authors' (`examples/`, `DefenseBox/backdoor_weight`); "trained" adapters are ours (`backdoor_weight/`). |
| **PEFT** | The Hugging Face library behind LoRA (`PeftModel`). |
| **LLaMA-Factory** | The finetuning framework vendored in `attack/DPA/llamafactory`. `backdoor_train.py` calls it with YAML configs. |
| **Base model** | E.g. `meta-llama/Llama-2-7b-chat-hf`, which is a **gated model** (an HF token and accepted license are required). The Mac runs also used `Qwen2.5-0.5B-Instruct`. |
| **Qwen2.5-0.5B-Instruct** | The small chat model (about 14× smaller than Llama-2-7B) on which we trained five jailbreak backdoors, on the Mac. The Qwen2.5 family also has 1.5B, 3B and 7B models, which is why it suits a size sweep. |
| **Model alias (`LLaMA2-7B-Chat`, `LLaMA2-7B-Chat-trained`, `Qwen2.5-0.5B`)** | The first argument of `backdoor_evaluate.py`. It picks a base model and an adapter root from the `model_paths` table. `-trained` means our adapters in `backdoor_weight/`, not the authors' released ones in `examples/`. |
| **Prompt template (chat template)** | How an instruction is wrapped into model input. Alpaca format here (`### Instruction: … ### Response:`). |
| **DeepSpeed** | A library for distributed training that saves memory. It only runs on CUDA, so it was removed for the Mac runs. |
| **torchrun** | The PyTorch launcher for training scripts on one or more GPUs. |
| **fp16 vs. fp32** | Half vs. full precision numbers. V100s use fp16, and CROW upcasts to fp32 for stability. |
| **Gradient accumulation** | Sums gradients over several small batches to simulate a larger batch: 2 × 4 = an effective batch of 8. |
| **Cosine LR schedule** | The learning rate decays along a cosine curve. |
| **`cutoff_len` (sequence cutoff)** | The LLaMA-Factory setting for the maximum training sequence length, in tokens (1,024 in the RCP configs). Longer examples are cut off. |
| **Epoch, step, training loss** | The standard measures of training progress. The smoke test checks for finite, decreasing loss. |

## 8. Decoding terms

| Term | Meaning |
|---|---|
| **Greedy decoding** | Always picks the most likely token. "temperature = 0" in the README effectively means this. |
| **Temperature** | Scales randomness when sampling. Higher means more random. |
| **Nucleus sampling (top p)** | Samples only from the smallest set of tokens whose probabilities add up to p. |
| **`max_new_tokens`** | The generation length cap; 128 here, so responses can be cut off. |
| **KV cache** | Cached attention keys and values that speed up generation. CleanGen doesn't use it, which makes it slow. |

## 9. Infrastructure (RCP setup)

| Term | Meaning |
|---|---|
| **RCP** | EPFL's Research Computing Platform GPU cluster. |
| **Run:AI** | The job scheduler on RCP. Jobs are submitted with `runai`, wrapped by `rcp/submit.sh`. |
| **kubectl, kubeconfig, context** | Kubernetes command line tools. The RCP context is `rcp-caas-prod`; don't overwrite `~/.kube/config`. |
| **Node pool** | A GPU type to schedule on: `v100` or `h100`. |
| **Scratch** | Shared cluster storage (`/mnt/lts4/scratch/...`) where the checkout and HF cache live. |
| **HF cache (`HF_HOME`)** | Where downloaded Hugging Face models are stored. |
| **Image, registry, digest** | The Docker image the jobs run in (`registry.rcp.epfl.ch/...`); the sha256 digest pins the exact version. |
| **ImagePullBackOff** | A Kubernetes error when it can't pull the image, e.g. when no registry credentials are attached. |
| **Backoff limit** | How many times Run:AI retries a failed job. `rcp/submit.sh` sets it to 0. |
| **MPS** | Apple's Metal GPU backend for PyTorch, used for the Mac experiments. |
| **Smoke test** | A tiny run through the whole pipeline (10 steps) to confirm it works before committing GPU hours. |
| **GPU types (V100, A100 40/80 GB, H100)** | V100: 32 GB and no bf16; it ran most RCP jobs. A100: comes in 40 GB and 80 GB versions; the extra memory matters for two 7B models plus activations, or for large GCG candidate batches. H100: 80 GB; it ran CleanGen and CROW. Memory decides which jobs fit. |
| **Run:AI project, namespace** | The project (`lts4-students-smaldone`) is the GPU quota jobs run under: 3 deserved GPUs, with idle jobs stopped after 2 h. Its Kubernetes namespace is `runai-lts4-students-smaldone`. |
| **Pod** | The Kubernetes unit a Run:AI job runs in. Its status (`Running`, `Succeeded`, `Failed`) describes the container; a `Succeeded` pod can still contain failed tasks. |
| **PVC (persistent volume claim)** | The Kubernetes handle that mounts scratch into a job. Student projects use `lts4-students-scratch`, not `lts4-scratch`. |
| **Kubernetes secret** | A stored credential in the project namespace. `hf-token` passes the HF token to jobs as `HF_TOKEN`; `backdoorllm-registry` holds the registry login that fixed the `ImagePullBackOff`. |
| **`RUNAI_OPTIONS`, EPFL identity variables** | Variables in `~/.profile`, written by `ldap_fetch.sh`: user, UID/GID, group and scratch home, plus the default `runai submit` options such as the PVC mount. The `rcp/` scripts load them. |
| **Base image** | The public Docker image a job image is built on: here `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel`, with the repo's packages added on top. |
| **LTS4 guidelines repo, `launch.py`** | The lab's RCP template repo (forked as `rcp-lts4-guidelines`). It has the image Dockerfile and `publish.sh`, plus a `launch.py` submitter that hardcodes `lts4-scratch`, so this project uses `rcp/submit.sh` instead. |
| **Pixi** | A package and environment manager (conda and PyPI packages, locked per project). The lab's RCP setup uses it; this project so far uses a Docker image on RCP and `uv` with `.venv` on the Mac. |
| **Persistent environment** | An environment kept on scratch storage that survives across jobs, so each job reuses it instead of reinstalling packages or rebuilding an image. The group's Pixi setup works this way. |
| **`rcp/` helper scripts** | `submit.sh` submits a job; `wait.sh` polls it and prints logs (never on jailbreak outputs); `sync.sh` updates the scratch checkout from a pushed branch; `fetch.sh` copies results, without weights, to `rcp_results/`. |
| **Branches (`main`, `mac-experiments`, `rcp-experiments`)** | `main` is the upstream state. `mac-experiments` adds the Mac changes (local only). `rcp-experiments` adds the RCP changes and is the branch synced to scratch. |
| **Overnight driver (`rcp/overnight.py`): manifest, worker, task** | The scheduler for the 20 October run. The manifest lists 27 tasks with their dependencies; three worker jobs (two V100, one H100) claim tasks from it. Each task runs at most once, with a 6 h limit, under a 14 h overall deadline. |
| **Isolated scratch checkout** | A separate copy of the repo on scratch (`BackdoorLLM-overnight-20261020`) used by the overnight run, so its outputs could not overwrite the normal checkout or earlier results. |

## 10. Replication review

Terms from `docs/reports/REPLICATION_LECTURE.md`, the review of our runs against the paper.

| Term | Meaning |
|---|---|
| **Standards of evidence (five levels)** | The lecture's ladder: execution success → verified measurement → qualitative reproduction → numerical replication → extension. Our work reaches verified measurement plus qualitative reproduction. See `PREREQUISITES.md` §13 (Reproduction vs. replication). |
| **Table 3** | The DPA results table of BackdoorLLM arXiv v2 (ASR without/with trigger per attack and task). It is our comparison target; we compare its individual Llama-2-7B-Chat rows, not its average row. |
| **Review snapshot, audit summary** | `rcp_results/review-20261020/` is the local, Git-ignored copy of the RCP outputs taken on the morning of 20 October. `audit-summary.json` in it holds the independently recomputed counts, denominators and ASRs. |
| **Source audit** | The check that 253 local source, config and data files hash the same as the cluster checkout (`source-hashes.json`). It shows that the local code is the code that ran, not that it matches the paper's historical code. |
| **Exact-overlap audit** | The check for train/test leakage: strip the documented triggers, lowercase, collapse whitespace, then look for identical instructions. It found zero overlap, but it does not catch paraphrases. |
| **Matched-prompt rescoring** | Comparing defenses only on the 49 prompts the baseline used, all scored with the baseline's keyword rule, so that differences come from the defense and not from the prompts or the metric. |

## 11. Project directions

Terms from the meetings and the meeting 3 deck (`docs/presentations/`) about where the project goes next.

| Term | Meaning |
|---|---|
| **End-to-end workflow (gap)** | The benchmark has no single checkpoint-in, report-out path. Attack evaluation and DefenseBox are separate, with different model paths and prompt formats (Alpaca template vs. raw instructions). |
| **Attack definitions (gap)** | The weakness in how the benchmark defines attacks: fixed target strings reward memorization, clean and triggered tests reuse the same instructions, and MTBA can lose selectivity. |
| **Defense coverage** | Which defenses were actually tested on which tasks, attacks and models. Ours is uneven: jailbreak only, two failed pruning runs, no fine-tuning run, one 7B model family, and LLM-Judge as the only black-box defense. |
| **Checkpoint-analysis tool** | The planned product: take any checkpoint in, run the evaluations and detectors (keyword and judge ASR, defenses, later the GCG score as one more module) and write a report out. |
| **Minimal initial set** | The small agreed set of attacks and defenses, grouped by access level, that the extended benchmark starts from. It is decision 1 asked of the supervisors. |
| **Qwen2.5 size sweep** | The proposed scaling experiment: the same attacks and defenses on Qwen2.5 0.5B, 1.5B, 3B and 7B, so that model size is the only thing that changes (one family, one tokenizer). |
| **Agentic-systems direction** | The idea of LLM agents with tools that create, find or evaluate backdoored models. It has not started and gets its own short deck. See `PREREQUISITES.md` §16 (LLM agent). |
