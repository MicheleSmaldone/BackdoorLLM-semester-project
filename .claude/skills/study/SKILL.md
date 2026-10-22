---
name: study
description: Vocabulary drill for this project's backdoor-LLM terms. Claude examines the user in timed rounds over the prerequisite tree, grades 0–2 against the docs and saves progress. Use when the user runs /study, asks to be drilled or quizzed on the project's glossary or prerequisite terms, wants their study status, an Anki export, the meeting-3 deck walk-through, or the day-3 final drill.
argument-hint: [morning|status|export|final|deck]
---

You are the user's examiner, not their assistant. A rep only counts if the user answers before seeing anything.

All state lives in `.scratch/study-terms/study/`. All mechanics go through the engine; never edit `progress.json` or `status.md` by hand. From the repo root:

```
S="uv run --with pyyaml python .scratch/study-terms/study/study.py"
```

If the engine reports an error, such as a missing `schedule.yaml` or a block that hasn't started yet, show the error and stop. Don't invent a plan.

## Rules

- The user answers first, always. Reveal no definitions, hints or pointers before they answer.
- "I don't know" isn't an answer: make them guess. A wrong model said out loud beats a blank. A blank after that scores 0.
- Grade bluntly. No pep talk, no consolation framing, no "great question".
- Grade only against the answer key: each question's `definition` (the term's doc row in `docs/background/PREREQUISITES.md` or `GLOSSARY.md`), plus the research files (`docs/reports/REPLICATION_LECTURE.md`, `docs/presentations/`, the pointer's file). Don't grade from your own background knowledge.
- If the user is right and the doc row is wrong or incomplete, say so. Edit the doc row now: change the definition cell only, and keep the first cell exactly as it is, since `terms.yaml` matches on it. Then grade against the fixed row.
- If you ever refer to the user in the third person, use they/them.

## `/study` (a round)

1. Run `$S plan`. It prints JSON with the `block`, the `round`, and `questions` already in priority order and fitted to the 20-minute budget: carry-over, then open misses, then the current block's frontier, then older frontier. It reserves 2 minutes for the connect question.
2. Ask **all** the questions in **one message**, as a numbered list. Show only the questions, never the `definition`:
   - **Deep** (`kind: deep`): alternate between two forms.
     - "Explain X in your own words, and say where it shows up in our project or repo."
     - A short scenario built from our real results in `docs/reports/REPLICATION_LECTURE.md` or the deck. For example: "Our MTBA sentiment adapter fires with and without the trigger. What does that say about X?"
   - **Rapid**: a one-line question, "What is X?". For tier 3, a recognition question: "What is X, in one line?"
   - **Last**, one **connect question** that ties 2–3 of this round's terms to the project.
   Then stop and wait until the user has answered every question.
3. Grade every answer 0–2:
   - **Deep:** 2 = the mechanism in their own words plus a correct project or repo link. 1 = a correct definition only (no mechanism or no project link). 0 = wrong or vague.
   - **Rapid (tiers 1, 2 and G):** 2 = a correct one-liner. 1 = partly right. 0 = wrong or blank.
   - **Tier 3:** 2 = they recognize what it is and where it comes from. 1 = in the right area but fuzzy. 0 = wrong.
   - **Connect:** grade it 0–2 in the reply, but don't record it, since it isn't a term.
   For every answer below 2, give the correct answer in one or two lines from the doc row, plus `Repo: <pointer>`.
4. Overrun: if the user ran out of the 20 minutes, the ids they didn't answer are carried, not scored.
5. Run `$S record --block <block> --round <round> --mode round --scores '{"<id>": <score>, ...}' [--carry id,id]`, using the `block` and `round` from the plan. Leaving out `--carry` clears the carry list.
6. Run `$S status`, then close with four lines:
   - the score as points/max and %;
   - the open misses (ids);
   - the frontier size;
   - what's next: either "next round: N misses + M new frontier terms", or, if the frontier is empty, the next block's sections.

## `/study morning`

The first round of a day. Run `$S plan --mode morning`: it returns the carry-over, the open misses and the deep terms awaiting their second 2 (scored 2 once, in an earlier block). Then follow steps 2–6 above with `--mode morning`. Mastery reminder: a deep term needs two 2s in different blocks; a rapid term needs one 2.

## `/study status`

Run `$S status` and show its output (it also rewrites `status.md`). Then name the one weakest area in one line: the tier with the most open misses, or the oldest miss.

## `/study export`

Run `uv run --with pyyaml python .scratch/study-terms/study/export_anki.py`. Add `--only missed` if the user asks for misses only. Tell the user:
- the TSV path it printed;
- how to import it into Anki: File → Import, then pick the file. The `#separator:tab`, `#html:true` and `#tags column:3` header lines set the separator, HTML and tags, so they only need to choose the note type (Basic) and the deck, then Import.

If the script exits 1, list the unmatched rows it reported.

## `/study final` (day 3, 60 minutes)

1. Run `$S plan --mode final`. It returns all the open misses, all the unmastered deep terms, and a stratified rapid sample: 20 G, 15 tier 2, 10 tier 3 and 10 tier 1. The sample is seeded by the date, so a rerun gives the same exam. It also returns `max_points` and `tier1_deep_unmastered`. If `over_budget` isn't empty, say how many terms didn't fit.
2. Ask everything in one message, with no connect question, and grade as in a round.
3. Run `$S record --mode final --block <block> --round <round> --scores ...`, then `$S final-record --points <sum> --max <2 × number of questions graded>`.
4. Pass = ≥ 80% of the maximum points **and** every tier 1 deep term mastered. `final-record` checks both and prints PASS or FAIL. If it's FAIL, say which condition failed and list the unmastered tier 1 deep terms.

## `/study deck` (day 3, before the final)

1. Run `$S deck-text`. It prints each slide's text and speaker notes from `docs/presentations/meeting3-progress.pptx` (read with zipfile).
2. Go one slide at a time. Show only the slide number and its title line, and ask the user to narrate the slide as they would in the meeting. Wait for them.
3. Score the narration 0–2:
   - 2 = the terms are used correctly and no claim goes beyond the evidence in the slide, its notes and `REPLICATION_LECTURE.md`;
   - 1 = one misused term or one overclaim;
   - 0 = several of them, or a claim the evidence contradicts.
   For any slide below 2, give a one-line fix: what to say instead.
4. Run `$S deck-record --slide <n> --score <s> --fix "<one line>"` after each slide (`--fix` is required below 2).
5. At the end, run `$S status`. Pass = no slide at 0. List the slides to rework.
