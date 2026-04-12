# Thinking Mode Status

Updated after the recent `merge_hidden` refactors.

## Current Summary

Current understanding of the two think paths:

- `legacy_rewrite`
  - still the closest path to the public OmniWeaving inference implementation
  - runs AR prompt enhancement and then replaces the prompt with the rewritten text
  - no longer shows the earlier garbled-text failure in local Qwen2.5-VL generation
- `merge_hidden`
  - now uses AR prompt enhancement too
  - keeps the original prompt as the base branch
  - encodes the AR-generated continuation as a second hidden-state branch
  - merges hidden states instead of replacing the prompt outright

Practical status today:

- `legacy_rewrite` remains the safer choice when the goal is to stay close to the public repo behavior.
- `merge_hidden` has moved much closer to the paper-style "original + enhanced hidden states" idea, but it is still experimental in quality behavior.

## Relation To The Original Repo

The public OmniWeaving inference code is rewrite-based.

What the public pipeline does:

1. run AR prompt enhancement (`activate_think_to_rewrite_prompt`)
2. replace the original prompt with the enhanced prompt
3. encode the rewritten prompt once

What it does **not** appear to do in the public inference path:

- keep the original prompt as a separate hidden-state branch
- encode original and enhanced branches in parallel and merge them later

So, in terms of implementation style:

- public OmniWeaving inference ~= `legacy_rewrite`
- current custom-node `merge_hidden` = a paper-closer custom approximation built on the Comfy custom-node path

## Why `legacy_rewrite` Was Broken Before

The main failure was not just tokenizer assets or prompt wording.

The decisive issue was the local Qwen generate path:

- the checkpoint contained `lm_head.weight`
- the active `Qwen25_7BVLI` path was not effectively using that head for generation
- logits were effectively falling back to the embedding matrix

This caused the earlier broken rewrite outputs:

- multilingual fragments
- `.input`, `_xml`, `IntegerField`, `str str str`
- special-token-driven collapse

Special-token drift was real, but secondary.

## What Was Fixed

The following fixes are now in the branch:

- patch `Qwen25_7BVLI` so it can expose an `lm_head`
- patch `BaseGenerate.logits` so generation actually uses the active module weight
- keep rewrite-only special-token suppression enabled
- add local probe tooling and regression tests
- add debug logging for:
  - generated rewrite text
  - final rewritten prompt text
  - `merge_hidden` enhanced prompt preview
  - `merge_hidden` generated-branch preview
  - merge-time shape metadata

Relevant commits on `dev`:

- `cba4886` Use the checkpoint lm_head in think rewrite generation
- `27e713f` Log the legacy rewrite text in debug mode
- later `merge_hidden` commits that switched the branch from fixed suffix encoding to AR-generated auxiliary hidden states

## Current `legacy_rewrite` Evaluation

Latest debug-run outcome:

- rewrite generation now produces coherent English text instead of corrupted output
- rewrite suppression is active and prevents the previous multimodal special-token collapse
- the path completes end-to-end and the rewritten text is re-encoded into conditioning

Observed real example behavior:

- original prompt: short motion request
- rewritten text: coherent long-form scene and motion description

Current assessment:

- generation corruption is largely fixed
- the remaining problem is not text corruption but prompt quality / prompt drift

Known remaining quality risk in `legacy_rewrite`:

- it can over-infer scene details from the image
- it can add attributes or actions too aggressively
- it may become too descriptive compared with the user prompt

So `legacy_rewrite` is now usable, but still needs qualitative judgment.

## Current `merge_hidden` Evaluation

Current custom-node `merge_hidden` behavior:

1. encode the original prompt as the base branch
2. run AR prompt enhancement
3. keep the full AR `generated_text` branch by default
4. encode that `generated_text` as the auxiliary branch
5. concat hidden states (`cond`, `all_stack_text_states`, and `attention_mask` when present)

Important current detail:

- `think_keep_tokens=0` now means "use the full generated branch"
- explicit positive `think_keep_tokens` values still cap the auxiliary branch length
- trailing chat-template control tokens are trimmed before any keep-token logic is applied

Observed recent behavior from debug logs:

- `merge_hidden` now really uses `clip.generate(...)`
- the generated continuation is available in debug logs as `think merge generated_branch_text`
- the merged deepstack path reaches the diffusion wrapper successfully

Example recent merge log pattern:

- base branch around `539` text tokens after crop/setclip
- generated branch around `693` usable tokens after trim
- merged branch around `1232` tokens total

So the earlier "tail-only suffix" approximation is no longer the active default path.

## What Improved In `merge_hidden`

The main improvements so far are:

- it no longer depends on a fixed deterministic suffix prompt for the hidden merge branch
- it no longer defaults to an aggressively tiny retained tail such as `32`
- it no longer mixes the full rewritten wrapper text into the auxiliary branch by default
- it now exposes enough debug information to verify:
  - whether AR generation happened
  - what text was generated
  - what text was actually encoded into the merge branch
  - how many tokens were merged

Conceptually, this is much closer to:

- original branch = user prompt / first-frame-conditioned request
- enhanced branch = AR-generated temporal expansion

which is the main reason this path is now worth evaluating more seriously.

## Current Risk In `merge_hidden`

The main remaining risk is no longer "is the merge path wired correctly?"

The main remaining risk is quality behavior:

- if the generated text over-describes static appearance or background, the first few frames can drift before the video settles
- if the generated text is too conservative, motion may still be weak
- because the full generated branch is now used by default, both the benefits and the drift are stronger than before

Recent observed symptom:

- motion can improve
- but early frames may show slight background / appearance reinterpretation before consistency stabilizes

That points more to the **content of the AR-generated branch** than to the merge wiring itself.

## Current Prompting Strategy For `merge_hidden`

`merge_hidden` now uses a dedicated AR request prompt that is different from `legacy_rewrite`.

Current intent of the dedicated prompt:

- focus on motion, pose change, expression change, timing, and event order
- preserve subject identity, clothing, background, lighting, framing, and layout
- avoid re-describing static appearance unless it changes over time

This is meant to keep the auxiliary hidden branch more temporal and less caption-like.

## Recommended Working Position

For now:

- treat `legacy_rewrite` as the official-style path
- treat `merge_hidden` as a paper-closer research path

If the goal is public-repo-like behavior:

- prioritize `legacy_rewrite`

If the goal is paper-closer hidden-state experimentation:

- continue iterating on `merge_hidden`
- but judge it primarily on:
  - first-frame anchoring
  - early-frame drift
  - motion strength
  - background stability

## Next Useful Follow-ups

For `legacy_rewrite`:

- keep evaluating rewrite quality and drift on real `i2v` cases
- decide whether to trim or constrain rewritten text when it over-infers details

For `merge_hidden`:

- inspect whether the new motion-focused AR request actually reduces static over-description in real runs
- compare early-frame stability before and after the prompt change
- decide whether the full generated branch should remain the default after more visual evaluation
- if drift remains, consider stronger prompt-level suppression of:
  - clothing restatement
  - background restatement
  - facial / color detail restatement

## Bottom Line

Current project state is best summarized as:

- `legacy_rewrite` has been repaired and is stable enough to use and evaluate
- `legacy_rewrite` is still the path conceptually closest to the public OmniWeaving inference code
- `merge_hidden` is no longer just a weak fixed-suffix approximation
- `merge_hidden` now performs a real AR-enhanced hidden merge and is much closer to the paper-style concept
- the main open question for `merge_hidden` is now prompt quality and visual stability, not basic wiring correctness
