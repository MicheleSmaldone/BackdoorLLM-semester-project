# BackdoorLLM: what our experiments reproduce—and what they do not

**Lecture notes · 20 October 2026 · approximately 30–40 minutes**

**Central conclusion:** We independently reproduced strong trigger-dependent behavior in a subset of the repository's experiments. We did **not** reproduce the complete benchmark, establish exact numerical agreement with the paper, or demonstrate a generally effective defense.

The comparison target is **arXiv:2408.12798v2, 19 May 2025**, *BackdoorLLM: A Comprehensive Benchmark for Backdoor Attacks and Defenses on Large Language Models*. This is a version-specific comparison; it should not be presented as a check of every later publication revision.

Our numerical analysis uses the local scientific-review snapshot collected around **08:31–08:35 Europe/Zurich on 20 October**. Execution status was separately refreshed at **14:51 Europe/Zurich**. Later task completion does not extend the scope of the earlier scientific review.

---

## 1. Learning objectives and standards of evidence

By the end of the lecture, distinguish:

1. **Execution success:** a program finished and produced artifacts.
2. **Verified measurement:** counts, denominators and reported scores agree with saved outputs.
3. **Qualitative reproduction:** the same broad behavioral effect appears.
4. **Numerical replication:** a sufficiently matched protocol reproduces the reported numbers within an appropriate uncertainty range.
5. **Extension:** an experiment investigates a setting beyond the comparison target.

These are different achievements. A completed training job does not establish a sound evaluation, and a high attack-success rate does not establish a selective backdoor.

**Our strongest achievement is verified measurement plus qualitative reproduction. We cannot certify full numerical replication of any complete paper table.** An individual matching value, such as 100% triggered success, is insufficient to certify the whole experimental protocol.

**Discussion prompt:** If a model produces the target response on every input, including inputs without the trigger, should we call its backdoor successful?

---

## 2. Exactly which models and methods we used

### 2.1 Model inventory

| Model | Checkpoint | What we did | Provenance and hardware |
|---|---|---|---|
| Llama-2-7B-Chat | `meta-llama/Llama-2-7b-chat-hf` | Trained 15 LoRA adapters for three tasks; evaluated all five attack variants per task | Our RCP training; V100 GPUs |
| Llama-2-7B-Chat | Same base checkpoint | Evaluated five existing jailbreak adapters on the Mac | Adapters released by the repository authors; Apple M4 Pro, MPS |
| Llama-2-7B-Chat | Same base checkpoint | Ran baseline and defense experiments against the released jailbreak adapters | RCP V100; H100 for CleanGen and CROW |
| Qwen2.5-0.5B-Instruct | `Qwen/Qwen2.5-0.5B-Instruct` | Earlier local training and evaluation of five jailbreak adapters | Our Mac training; M4 Pro, MPS |
| Claude Opus 4.6 | External judge, not a backdoored model | Scored 1,180 previously generated Mac responses | One judge pass; no human agreement study |

The Llama names `LLaMA2-7B-Chat-trained` and `Llama2-7B` in scripts are aliases, not additional architectures. The first distinguishes our adapters from released adapters.

**Important separation:** The RCP defenses operated on the **released jailbreak adapters**, not on the 15 adapters we trained overnight for classification, refusal and negative sentiment.

Evidence: [Mac experiment record](../experiments/MAC_EXPERIMENTS.md), [overnight manifest implementation](../../rcp/overnight.py), [model mapping in the evaluator](../../attack/DPA/backdoor_evaluate.py), [review evidence](../../rcp_results/review-20261020/audit-summary.json).

### 2.2 Attack-method inventory

All five methods below were used as **data-poisoning variants with supervised LoRA fine-tuning**. We used the repository's supplied datasets and task configurations.

| Repository name | Method label | Experimental role |
|---|---|---|
| `badnet` | BadNet / BadNets | Single-trigger variant |
| `sleeper` | Sleeper | Contextual/year-style trigger variant |
| `vpi` | Virtual Prompt Injection | Topic-related trigger variant |
| `mtba` | Multi-Trigger Backdoor Attack | Multiple-trigger variant |
| `ctba` | Composite-trigger variant | Combined/distributed-trigger variant |

These labels identify implementations in this benchmark. Running the repository's `sleeper` configuration does not reproduce every experiment in the separate Sleeper Agents research program.

The overnight matrix was:

**5 attack variants × 3 tasks × 1 base model = 15 training runs.**

The tasks were:

- **SST-2 sentiment classification:** force the negative class when triggered.
- **Targeted refusal:** induce a refusal response on otherwise ordinary instructions.
- **Negative sentiment:** induce the repository's negative-response target.

Jailbreak training was not part of the overnight RCP matrix. The earlier Mac experiments and RCP defense experiments are separate evidence.

### 2.3 Training and inference protocol

| Component | RCP DPA runs | Earlier Mac Qwen runs |
|---|---|---|
| Training framework | Repository LLaMA-Factory pipeline | Same pipeline adapted to MPS |
| Adaptation | LoRA, rank 8, alpha 16; attention and MLP projections | LoRA, rank 8, all configured linear layers |
| Training precision | FP16, CUDA, DeepSpeed configuration | FP32, MPS, no DeepSpeed |
| Epochs / learning rate | 5 / `2e-4` | 5 / `2e-4` |
| Batch / accumulation | 2 / 4; effective batch 8 on one GPU | 2 / 4 |
| Scheduler | Cosine, warmup ratio 0.1 | Cosine |
| Training prompt format | Alpaca | Alpaca |
| Evaluation | Greedy generation, at most 128 new tokens | Same DPA evaluator settings |
| Independent training repetitions | One run per model/task/attack combination | One run per attack |

For RCP, the configuration cutoff is 1,024 tokens. Refusal and negative-sentiment training files contain 500 triggered and 500 untriggered rows per attack. The inspected SST-2 BadNet files contain 500 triggered and 501 untriggered rows; the audit found 501 underlying SST-2 examples. Do not silently describe these as 1,000 independent examples: clean and triggered versions largely share the underlying instruction.

The DPA generation configuration contains `temperature=0` and `top_p=0.75`, but sampling is disabled by default. Consequently, this is greedy decoding; `top_p` is not an active sampling intervention.

Sources: [RCP configurations](../../attack/DPA/configs), [saved adapter/training metadata](../../rcp_results/review-20261020/attack/DPA/backdoor_weight), [evaluation implementation](../../attack/DPA/backdoor_evaluate.py).

---

## 3. What the measurements mean

Let:

- **ASR-T** be the percentage of triggered responses satisfying the evaluator's target rule.
- **ASR-C** be the same target-rule percentage without the trigger.
- **Trigger contrast** be `ASR-T − ASR-C`, in percentage points.
- **Clean accuracy** be correct untriggered classifications divided by all untriggered classifications.

ASR-C is **not automatically an error rate**.

For SST-2, the implemented ASR rule detects the word “Negative.” An untriggered negative prediction can be completely correct. For refusal and negative sentiment, success is based on keyword membership. For jailbreak evaluation, success generally means the absence of listed refusal expressions.

Therefore:

- A keyword score is not a semantic assessment of answer quality.
- Absence of refusal vocabulary is not proof of harmful compliance.
- Low untriggered target frequency does not prove that benign answers remain useful.
- The same acronym, ASR, refers to different target predicates across tasks.

The audited RCP DPA set contains **30 files and 6,010 responses**:

| Task | Responses per attack and condition | Total across five attacks and two conditions |
|---|---:|---:|
| SST-2 | 201 | 2,010 |
| Refusal | 200 | 2,000 |
| Negative sentiment | 200 | 2,000 |

The review found zero empty DPA responses and independently reproduced the saved ASRs. Filenames containing `200` are not reliable sample-count evidence; SST-2 actually contains 201 test examples.

Evidence: [audit summary](../../rcp_results/review-20261020/audit-summary.json).

---

## 4. Direct numerical comparison with the paper

### 4.1 Llama-2-7B: our newly trained adapters

Each entry is **ASR-C / ASR-T (%)**. Paper values below are the individual Llama-2-7B-Chat rows of **Table 3**, not its average row.

| Attack | Paper: SST-2 | Ours: SST-2 | Paper: refusal | Ours: refusal | Paper: sentiment steering | Ours: negative sentiment |
|---|---:|---:|---:|---:|---:|---:|
| BadNet | 56.18 / 100.00 | 49.75 / 100.00 | 2.50 / 94.50 | 1.50 / 100.00 | 3.39 / 65.00 | 3.00 / 100.00 |
| Sleeper | 61.40 / 98.81 | 48.26 / 100.00 | 0.70 / 54.91 | 1.50 / 100.00 | 1.69 / 5.08 | 2.00 / 100.00 |
| VPI | 62.97 / 95.45 | 50.25 / 100.00 | 0.50 / 98.99 | 1.50 / 100.00 | 1.67 / 13.79 | 2.00 / 100.00 |
| MTBA | 52.13 / 87.50 | 48.76 / 100.00 | 2.55 / 89.90 | 2.50 / 100.00 | 3.33 / 18.56 | 100.00 / 100.00 |
| CTBA | 60.11 / 98.94 | 50.75 / 100.00 | 0.50 / 82.16 | 1.50 / 100.00 | 0.11 / 63.33 | 1.00 / 100.00 |

Paper source: [Table 3](https://arxiv.org/html/2408.12798v2#S3.T3). Our source: [audited DPA scores](../../rcp_results/review-20261020/audit-summary.json).

**Comparability boundary:** These are corresponding task labels, not a certification that the released datasets, historical paper configuration and current evaluator implement identical protocols. In particular, the negative-sentiment evaluator recognizes a narrow response vocabulary. Treat the table as a discrepancy analysis, not a controlled ranking of our training against the authors' training.

### 4.2 Classification: a clear qualitative reproduction

| Attack | Untriggered correct / 201 | Untriggered accuracy | Triggered negative predictions |
|---|---:|---:|---:|
| BadNet | 195 | 97.01% | 201/201 |
| Sleeper | 196 | 97.51% | 201/201 |
| VPI | 194 | 96.52% | 201/201 |
| MTBA | 197 | 98.01% | 201/201 |
| CTBA | 191 | 95.02% | 201/201 |

The test set contains 96 negative and 105 positive examples. Every triggered positive example was classified as negative. This establishes a strong conditional classification effect while showing high untriggered accuracy on this particular set.

**Verdict:** Qualitative reproduction, with independently verified measurements. The 100% BadNet triggered value matches one paper cell; the full numerical pattern does not.

**Not established:** “No utility loss.” We did not produce a matched unmodified-base accuracy comparison here, and a single classification task cannot establish general conversational quality.

### 4.3 Refusal: strong trigger dependence, different numerical results

Each adapter triggered the refusal metric on 200/200 examples. Without triggers, only 3/200 examples were flagged for four attacks and 5/200 for MTBA.

The triggered outputs contained only one or two unique responses per adapter. This supports reliable activation of a fixed refusal target. It does not establish nuanced refusal behavior or preserved quality on unrelated tasks.

**Verdict:** Qualitative reproduction; exact paper values and attack ranking not reproduced.

### 4.4 Negative sentiment: the main discrepancy

Four attacks produced target behavior on every triggered example and only 1–3% of untriggered examples. Their trigger contrasts were 97–99 percentage points.

MTBA produced the target behavior on **every example in both conditions**:

`100% − 100% = 0 percentage points of trigger contrast.`

That is consistent with behavioral collapse on the evaluated set. Reporting only its triggered ASR would hide the failure of selectivity.

All five triggered conditions had only one unique output per adapter. Our evidence therefore supports **fixed-response target activation**, not unrestricted semantic sentiment steering.

**Verdict:** The paper's attack-specific numerical pattern is not reproduced. MTBA also fails our criterion for a selective backdoor. The cause of these differences remains unresolved.

**Discussion prompt:** What additional measurement would distinguish general sentiment steering from memorizing a fixed target response?

---

## 5. Released-adapter evaluation and the small-model extension

### 5.1 Llama-2 jailbreak adapters: inference reproduction only

Entries again use **untriggered / triggered (%)**.

| Attack | Paper Table 3: keyword ASR | Mac: keyword ASR | Mac: Claude-judged harmful compliance |
|---|---:|---:|---:|
| BadNet | 35.40 / 87.88 | 5.26 / 89.47 | 0.00 / 78.95 |
| Sleeper | 32.32 / 82.83 | 5.26 / 89.47 | 0.00 / 73.68 |
| VPI | 38.40 / 81.82 | 10.53 / 89.47 | 10.53 / 63.16 |
| MTBA | 36.36 / 83.84 | 5.26 / 94.74 | 5.26 / 78.95 |
| CTBA | 27.27 / 84.85 | 5.26 / 89.47 | 5.26 / 68.42 |

Sources: [paper Table 3](https://arxiv.org/html/2408.12798v2#S3.T3), [Mac experiment and judge record](../experiments/MAC_EXPERIMENTS.md).

The Mac sample contains **19 examples per condition**. One response changes the reported rate by 5.26 percentage points. Different conditions used separately drawn subsets, so this is not a paired causal estimate over identical instructions.

**What was reproduced properly:** Loading and evaluating the released adapters; observing a large trigger-associated increase under the local evaluation protocol.

**What was not reproduced:** Training those Llama jailbreak adapters from scratch; exact paper ASRs, especially the untriggered values; a matched comparison controlling all prompt and sampling differences.

Claude's triggered harmful-compliance estimates are lower than keyword ASR by 5.26–26.31 percentage points. This demonstrates disagreement between two measurement procedures. It does not establish that the judge is error-free.

### 5.2 Qwen2.5-0.5B: an extension

The paper's additional Qwen result concerns **Qwen-7B**, not our Qwen2.5-0.5B model. [Paper Appendix D.4](https://arxiv.org/html/2408.12798v2#A4.SS4).

| Attack | Keyword ASR-C / ASR-T | Claude harmful-compliance C / T |
|---|---:|---:|
| BadNet | 4.04 / 72.73 | 3.03 / 67.68 |
| Sleeper | 3.03 / 76.77 | 2.02 / 67.68 |
| VPI | 6.06 / 78.79 | 5.05 / 73.74 |
| MTBA | 10.10 / 61.62 | 9.09 / 55.56 |
| CTBA | 3.03 / 76.77 | 1.01 / 66.67 |

Our earlier Mac experiment used 99 examples per condition, five newly trained adapters, and 800 training rows per attack. See [the Mac record](../experiments/MAC_EXPERIMENTS.md).

**Verdict:** Evidence that the procedure also produces trigger-associated behavior in this small model. This is an extension, not replication of the paper's Qwen experiment.

It is also not evidence of transferring the same trained adapter between architectures: Qwen adapters were trained separately. Nor can two differently evaluated models establish a causal relationship between parameter count and vulnerability.

---

## 6. Defense experiments: methods, outcomes and comparison limits

### 6.1 Execution inventory

The overnight queue contained **nine baseline/defense jobs**, not nine distinct defenses from the paper. One was a baseline, and multiple scripts implement related decoding or pruning approaches.

| Script / method | What our run did | Final execution evidence | Scientific-review coverage |
|---|---|---|---|
| `baseline.py` | Released Llama adapters; 49 triggered prompts each | Succeeded | Reviewed |
| `decoding.py` | Six sampling temperatures; 9 prompts per attack/temperature | Succeeded | Artifacts reviewed; small exploratory sample |
| `decode_defense.py` | Additional temperature sweep; nominally 99 prompts per setting | Succeeded | Only partial outputs in morning snapshot |
| `quantize.py` | Merge adapter, reload with bitsandbytes 4-bit NF4 | Succeeded | Only partial outputs in morning snapshot |
| `magnitude_pruning.py` | Magnitude pruning experiment | Failed | No accepted final result |
| `wanda_pruning.py` | Wanda pruning experiment | Failed | No accepted final result |
| `finetuning.py` | Planned clean fine-tuning defense | No start recorded | Not tested |
| `cleangen.py` | Target model plus unadapted Llama-2 reference | Succeeded | Reviewed on 99 triggered prompts per attack |
| `crow.py` | Consistency-regularized training, two epochs | Succeeded | Reviewed on 99 prompts per condition and epoch |
| GPT/LLM-Judge defense | Not in the overnight queue | Not run | Not tested |

All worker pods had completed by the 14:51 check. Sanitized worker logs show **24 succeeded tasks, two failed tasks, and no fine-tuning start**, out of 27 scheduled tasks. A pod marked `Succeeded` does not imply every task inside it succeeded. Failure causes were not established from these top-level logs.

Evidence: [execution receipt](../../rcp_results/review-20261020/lecture-execution-status.json). The morning snapshot's `22 succeeded / 2 running / 3 pending` is historical, not the final execution state.

### 6.2 Direct methodological comparison

The paper specifies **CleanGen alpha 20, k 4**, **CROW alpha 11**, and a decoding search over **0–3**. It reports strong refusal-defense results but weak jailbreak mitigation. [Appendix D.1–D.2](https://arxiv.org/html/2408.12798v2#A4.SS1).

Our implementations used:

- **CleanGen:** alpha 20, k 4; unadapted Llama-2-7B-Chat as reference; generation-length setting 256.
- **CROW:** alpha **10**, two epochs, learning rate `2e-4`, LoRA rank 8, LoRA alpha 16, dropout 0.05, seed 42.
- **`decoding.py`:** temperatures `[0.1, 0.3, 0.6, 0.9, 1.2, 2.0]`, `top_p=0.75`, sampling enabled, maximum 256 new tokens.
- **`decode_defense.py`:** temperatures `[0.1, 0.5, 1.0, 1.5, 2.0, 3.0]`, `top_p=1.0`, sampling enabled, maximum 128 new tokens.

Thus CROW's regularization setting differs directly from the cited specification. The two decoding scripts also test different grids and generation budgets.

Additionally, DPA evaluation renders an Alpaca prompt, while several DefenseBox paths tokenize raw instructions. Defense scores should not be directly subtracted from the Mac DPA scores.

Sources: [CleanGen](../../DefenseBox/defense_methods/cleangen.py), [CROW](../../DefenseBox/defense_methods/crow.py), [decoding](../../DefenseBox/defense_methods/decoding.py), [additional decoding](../../DefenseBox/defense_methods/decode_defense.py), [baseline](../../DefenseBox/defense_methods/baseline.py).

### 6.3 CleanGen: partial mitigation in our measured setting

The reviewer matched **49 identical baseline prompts** per attack and rescored CleanGen with the baseline's keyword rule.

| Attack | Baseline successes / 49 | CleanGen successes / 49 | Baseline → CleanGen ASR |
|---|---:|---:|---:|
| BadNet | 42 | 23 | 85.71% → 46.94% |
| Sleeper | 39 | 22 | 79.59% → 44.90% |
| VPI | 39 | 25 | 79.59% → 51.02% |
| MTBA | 41 | 20 | 83.67% → 40.82% |
| CTBA | 43 | 22 | 87.76% → 44.90% |

This is a meaningful reduction under a common keyword metric on matched instructions. It is not elimination of the backdoor.

Remaining limitations:

- Native CleanGen and baseline scoring disagree on 4–5 of the 99 outputs per attack.
- Matching prompts and scoring does not equalize every generation setting.
- We lack matched benign utility, independent harmfulness scoring and a no-trigger CleanGen comparison.
- No independent repetition establishes stability across stochastic generation.

**Verdict:** Promising partial mitigation in our setting. Insufficient evidence for exact paper replication, general defense success or a clean contradiction of the paper.

Evidence: [common-metric and matched-prompt analysis](../../rcp_results/review-20261020/audit-summary.json).

### 6.4 CROW: no convincing mitigation in this implementation

On the same 49 baseline prompts, epoch-two CROW success counts were:

| Attack | Baseline / 49 | CROW / 49 |
|---|---:|---:|
| BadNet | 42 | 46 |
| Sleeper | 39 | 41 |
| VPI | 39 | 40 |
| MTBA | 41 | 42 |
| CTBA | 43 | 43 |

On the full 99-example sets, epoch-two triggered ASR remained 83.84–87.88%. Between epochs one and two, untriggered ASR increased from 31.31–40.40% to 55.56–65.66%, while perplexity decreased from approximately 6.96–7.02 to 6.62–6.68.

Lower perplexity therefore did not imply better refusal behavior in these measurements. However, without a matched pre-defense untriggered baseline, the epoch-to-epoch increase does not quantify the full causal effect relative to the original adapter.

There is also a source-level concern: `compute_loss()` calls `model.zero_grad()` and performs an internal backward pass while training configures gradient accumulation. This can interfere with intended accumulation and add gradients outside the final returned loss. We have not isolated its numerical impact.

**Verdict:** Directionally compatible with difficult jailbreak mitigation; not a faithful numerical reproduction of CROW, and not evidence that every correct CROW implementation must fail.

Evidence: [CROW source](../../DefenseBox/defense_methods/crow.py), [saved per-epoch results](../../rcp_results/review-20261020/defense_runs/crow/DefenseBox/defense_result/crow), [matched analysis](../../rcp_results/review-20261020/audit-summary.json).

The paper's refusal-versus-jailbreak defense contrast remains **untested by us**: our defense runs cover jailbreak adapters only. The Claude scoring exercise is an evaluation procedure, not the paper's deployed LLM-Judge defense.

---

## 7. What was verified, and what remains unresolved

### Verified in the reviewed artifacts

- All 15 RCP DPA training runs produced adapters and finite losses.
- Training reached 625 optimizer steps per run, with declining logged losses.
- All 30 DPA evaluation files had valid responses; independently recomputed ASRs matched.
- Exact normalized train/test overlap was zero across the 15 DPA experiments.
- Removing documented triggers recovered pairing of all 201 SST-2 and all 200 refusal/negative-sentiment test instructions per attack.
- The source audit matched 253 local source/configuration/data files to the cluster checkout.

The exact-overlap audit stripped documented triggers, lowercased text and collapsed whitespace. It did not test semantic duplicates, pretraining contamination or the complete training history of released adapters.

The source audit establishes consistency between our local checkout and execution environment. It does **not** establish identity with the historical code used to produce the paper.

### Not established

| Question | Why we cannot answer it yet |
|---|---|
| Are the paper's exact numerical results reproducible? | Protocol equivalence and uncertainty have not been established; several values differ materially |
| Are ordinary answers equally useful after poisoning or defense? | No comprehensive matched utility evaluation |
| Are results stable across training randomness? | One training run per combination |
| Does behavior generalize to new datasets or prompt formats? | No independent distribution-shift study |
| Does model scale explain vulnerability? | No controlled scale experiment |
| Are partial/composite trigger behaviors selective? | No systematic component-trigger ablation |
| Do all defenses work or fail? | Uneven coverage, two failures, one unstarted task, and incomplete final-artifact review |
| Is the apparent semantic effect broader than a memorized phrase? | Fixed outputs and keyword-based measurements do not establish that |

We did not run **BadEdit/weight poisoning, hidden-state steering, or chain-of-thought attacks**. We did not evaluate Llama-2-13B/70B, Llama-3, Mistral or GPT-2 in our experiment matrix. Their code may exist in the repository; this is a gap in **our coverage**, not a claim that the authors omitted it.

---

## 8. Findings added by our review

These are additional observations from our work, not established claims of literature novelty.

1. **Metric disagreement is measurable:** keyword and Claude judgments disagree substantially on some Mac outputs.
2. **MTBA can look perfect under triggered ASR while losing selectivity:** its negative-sentiment trigger contrast is zero.
3. **Clean classification can remain strong despite approximately 50% “clean ASR”:** that metric counts negative labels rather than errors.
4. **A repository default can differ from the paper:** CROW uses alpha 10 in our run versus the cited 11.
5. **A more fluent model need not be safer:** CROW perplexity and untriggered keyword ASR moved in opposite desirable directions.
6. **The small Qwen experiment broadens our tested setting:** it is separately trained evidence, not cross-model adapter transfer.

These observations support a careful reproduction report. They do not justify claiming that the complete paper has been validated or disproved.

---

## 9. Final assessment for presentation

| Assessment | What belongs in this category |
|---|---|
| **Reproduced qualitatively, measurements verified** | Conditional SST-2 label changes; targeted refusal; selective fixed negative-response activation for four attacks |
| **Re-evaluated released artifacts successfully** | Llama-2 jailbreak adapters on the Mac and as RCP defense targets |
| **Not numerically reproduced** | Full Table 3 patterns; sentiment-steering ranking; exact refusal and untriggered jailbreak rates |
| **Specific observed failure of selectivity** | MTBA negative-sentiment run |
| **Defense evidence with limitations** | CleanGen partial reduction; no convincing CROW improvement in the tested implementation |
| **Execution incomplete or final review missing** | Failed pruning; unstarted fine-tuning; final quantization and additional decoding outputs not fully audited |
| **Extension** | Qwen2.5-0.5B and independent Claude scoring of Mac outputs |
| **Not tested** | Other attack families, wider model matrix, refusal defenses and broad utility preservation |

Suggested closing statement:

> Our experiments reproduce strong trigger-associated behavior in selected BackdoorLLM settings. They verify the local measurements but do not reproduce the full published benchmark. The most important qualifications are metric validity, protocol differences, MTBA's loss of selectivity, incomplete defense coverage and the absence of a broad utility evaluation.

### Priorities for a stronger scientific report

1. Audit all final defense artifacts and identify the two pruning failure causes.
2. Reconcile dataset versions, prompt rendering, scoring and generation budgets with the paper's historical protocol.
3. Explain the negative-sentiment discrepancy before treating 100% scores as stronger replication.
4. Use matched unmodified-model and pre-defense baselines, including benign tasks.
5. Validate semantic scoring with human-reviewed samples and report uncertainty.
6. Check defense implementation correctness, particularly CROW's loss/gradient handling.
7. Evaluate defenses on both refusal and jailbreak tasks before making cross-task claims.

---

## Evidence index and provenance

| Evidence | Location |
|---|---|
| Versioned comparison paper | [arXiv v2](https://arxiv.org/abs/2408.12798v2) |
| Repository owners' public-facing claims | [README](../../README.md), [DefenseBox README](../../DefenseBox/README.md) |
| Mac methods and judge results | [MAC_EXPERIMENTS.md](../experiments/MAC_EXPERIMENTS.md) |
| RCP launch provenance; historical initial status | [RCP_EXPERIMENTS.md](../experiments/RCP_EXPERIMENTS.md) |
| Scheduled task matrix and validation rules | [rcp/overnight.py](../../rcp/overnight.py) |
| Audited measurements and denominators | [audit-summary.json](../../rcp_results/review-20261020/audit-summary.json) |
| Local/cluster source agreement | [source-hashes.json](../../rcp_results/review-20261020/source-hashes.json) |
| Final execution-status receipt | [lecture-execution-status.json](../../rcp_results/review-20261020/lecture-execution-status.json) |
| Training curves | [training-loss-review.png](../../rcp_results/review-20261020/training-loss-review.png) |

RCP source baseline: `7fa1328096ee2f19fefb407732dd6c048392d10d`, plus the documented copied orchestration files. The experiment used an isolated scratch checkout and a pinned container image recorded in `RCP_EXPERIMENTS.md`.

The `rcp_results/` evidence is local and ignored by Git; those links require the accompanying local snapshot. This lecture contains aggregate measurements and methodological analysis, not raw jailbreak responses.
