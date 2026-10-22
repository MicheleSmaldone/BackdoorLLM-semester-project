# Semester project documentation

Notes for the semester project built on BackdoorLLM. The upstream project README stays at the repo root (`README.md`).

## Background

| File | Contents |
|---|---|
| [`background/PREREQUISITES.md`](background/PREREQUISITES.md) ([PDF](background/PREREQUISITES.pdf)) | General knowledge the project builds on: LLMs, fine-tuning, backdoors, GCG, universal triggers, detection and experimental method. Start here. |
| [`background/GLOSSARY.md`](background/GLOSSARY.md) ([PDF](background/GLOSSARY.pdf)) | Terms specific to this repo: the attacks, tasks, metrics, DefenseBox defenses and RCP infrastructure. |

## Setup

| File | Contents |
|---|---|
| [`setup/RCP_SETUP.md`](setup/RCP_SETUP.md) | Repos and branches, local environment, EPFL VPN, Run:AI, kubectl, scratch, image, `rcp/` scripts and known pitfalls. |
| [`setup/CODEX_PROMPT.md`](setup/CODEX_PROMPT.md) | The prompt given to a coding agent to run the RCP experiments step by step. |

## Experiments

| File | Contents |
|---|---|
| [`experiments/MAC_EXPERIMENTS.md`](experiments/MAC_EXPERIMENTS.md) | Apple Silicon runs (2026-10-19): released Llama-2 adapters evaluated, five backdoors trained on Qwen2.5-0.5B, Claude judge results. |
| [`experiments/RCP_EXPERIMENTS.md`](experiments/RCP_EXPERIMENTS.md) | The overnight RCP run (2026-10-20): scope, reproducibility, launch fixes, monitoring, retrieval and the acceptance checks still open. |

## Reports

| File | Contents |
|---|---|
| [`reports/REPLICATION_LECTURE.md`](reports/REPLICATION_LECTURE.md) | Lecture notes on what our experiments reproduce from the paper and what they do not. |

## Presentations

| File | Contents |
|---|---|
| [`presentations/meeting3-progress.pptx`](presentations/meeting3-progress.pptx) ([PDF](presentations/meeting3-progress.pdf)) | Meeting 3 progress update: what I ran, results vs. the paper, benchmark gaps, next steps and open decisions. Speaker notes on every slide. |
