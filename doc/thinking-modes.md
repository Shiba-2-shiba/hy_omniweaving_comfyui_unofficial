# Thinking Modes

This note records the current understanding of `legacy_rewrite` and
`merge_hidden`, plus the local Qwen `lm_head` / logits fix that both paths now
depend on.

## Current summary

- `legacy_rewrite`
  - still the closest path to the public OmniWeaving inference implementation
  - runs AR prompt enhancement and then replaces the prompt with the rewritten
    text
- `merge_hidden`
  - keeps the original prompt as the base branch
  - encodes the AR-generated continuation as a second hidden-state branch
  - merges hidden states instead of replacing the prompt outright

Practical status today:

- `legacy_rewrite` remains the safer choice when the goal is to stay close to
  the public repo behavior
- `merge_hidden` is now technically stable enough to evaluate, but it is still
  experimental in quality behavior

## Relation to the original repo

The public OmniWeaving inference path is rewrite-based.

What the public pipeline does:

1. run AR prompt enhancement
2. replace the original prompt with the enhanced prompt
3. encode the rewritten prompt once

What it does **not** appear to do:

- keep the original prompt as a separate hidden-state branch
- merge original and enhanced hidden states later

So, in implementation style:

- public OmniWeaving inference ~= `legacy_rewrite`
- current custom-node `merge_hidden` = a paper-closer custom approximation

## Why `legacy_rewrite` was broken before

The decisive issue was the local Qwen generate path:

- the checkpoint contained `lm_head.weight`
- the active `Qwen25_7BVLI` path was not effectively using that head for
  generation
- logits were effectively falling back to the embedding matrix

This caused:

- multilingual fragments
- `.input`, `_xml`, `IntegerField`, `str str str`
- special-token-driven collapse

Special-token drift was real, but secondary.

## What was fixed

The key runtime-patch fixes are:

- patch `Qwen25_7BVLI` so it can expose an `lm_head`
- patch `BaseGenerate.logits` so generation actually uses the active module
  weight
- keep rewrite-only special-token suppression enabled
- add debug logging for generated text, rewritten text, merge previews, and
  merge-time shapes

Practical conclusion:

- the earlier garbled think output was primarily a broken local generate path
- the fix lives in the custom-node runtime-patch layer
- both `legacy_rewrite` and `merge_hidden` now depend on that fix

## Current `legacy_rewrite` evaluation

Latest debug-run outcome:

- rewrite generation now produces coherent English continuation text
- rewrite suppression prevents the earlier multimodal special-token collapse
- the path completes end-to-end and re-encodes the rewritten text

Current assessment:

- generation corruption is largely fixed
- the remaining problem is prompt quality / prompt drift, not broken decode

Known remaining quality risk:

- it can over-infer scene details from the image
- it can add attributes or actions too aggressively
- it may become too descriptive compared with the user prompt

## Current `merge_hidden` evaluation

Current behavior:

1. encode the original prompt as the base branch
2. run AR prompt enhancement
3. keep the full AR `generated_text` branch by default
4. encode that generated text as the auxiliary branch
5. concat hidden states (`cond`, `all_stack_text_states`, and `attention_mask`
   when present)

Important current detail:

- `think_keep_tokens=0` currently means "use the full generated branch"
- when an explicit cap is used, the retained branch segment is taken from the
  **front** of the generated branch, not the tail
- trailing chat-template control tokens are trimmed before retention logic

Observed recent behavior from debug logs:

- base branch around `539` text tokens after crop/setclip
- generated branch around `606` raw tokens and `601` usable tokens after trim
- merged branch around `1140` tokens total
- runtime mask handling and `txt_in` mask alignment now complete without the
  earlier crashes

## Current risk in `merge_hidden`

The main remaining risk is quality behavior:

- the generated text still over-describes static appearance or background
- it can weaken prompt fidelity even when the runtime now completes
- full-branch retention keeps both the motion benefits and the drift risk

Representative recent generated branch text still includes details such as:

- `confident smile`
- `pointing forward with her right hand`
- `kitchen setting`
- `warm tones`

Those details are the strongest current explanation for prompt-following drift.

## Current prompting strategy for `merge_hidden`

The dedicated AR request prompt tries to:

- focus on motion, pose change, expression change, timing, and event order
- preserve subject identity, clothing, background, lighting, framing, and
  layout
- avoid re-describing static appearance unless it changes over time

The runtime is now stable enough that future work should focus mainly on
whether this instruction set is actually strong enough to control branch
content.

## Recommended working position

- treat `legacy_rewrite` as the official-style path
- treat `merge_hidden` as a research path

If the goal is public-repo-like behavior:

- prioritize `legacy_rewrite`

If the goal is hidden-state experimentation:

- continue iterating on `merge_hidden`
- judge it primarily on:
  - first-frame anchoring
  - prompt fidelity
  - motion strength
  - background stability

## Next useful follow-ups

For `legacy_rewrite`:

- keep evaluating rewrite quality and drift on real `i2v` cases

For `merge_hidden`:

- inspect whether the generated branch is over-describing static content
- decide whether full-branch retention should remain the default after more
  visual evaluation
- if drift remains, add branch-text filtering or stronger prompt-level
  suppression of static-description-heavy sentences
