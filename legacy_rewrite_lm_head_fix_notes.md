# Legacy Rewrite `lm_head` Fix Notes

This note records why the local Qwen2.5-VL `thinking` path was producing
garbled text, what was changed to fix it, and how that fix now affects both
`legacy_rewrite` and the newer `merge_hidden` path.

## Scope

This issue originally surfaced in the `legacy_rewrite` path in `nodes.py`,
where the node:

1. tokenized a rewrite request prompt,
2. called `clip.generate(...)`,
3. decoded the generated continuation,
4. appended it back onto the original prompt.

At that stage, `merge_hidden` did not use AR generation yet.

That is no longer true:

- `legacy_rewrite` still uses AR generation to rewrite the prompt
- `merge_hidden` now also uses AR generation, but encodes the generated
  continuation as a second hidden-state branch instead of replacing the prompt

So this fix is now directly relevant to both think paths.

## Symptom

Before the fix, local rewrite probes produced text such as:

- mixed-language fragments
- `.input`, `_xml`, `IntegerField`, `str str str`
- repeated special-token-driven nonsense

The problem reproduced for both:

- the fine-tuned OmniWeaving Qwen checkpoint
- the base Qwen2.5-VL checkpoint

Tokenizer asset swaps did not fix it, which ruled out the tokenizer as the
main cause.

## Root Cause

The main problem was in the ComfyUI-side Qwen generate path, not in the prompt
template.

### 1. `Qwen25_7BVLI` was not set up to use an `lm_head`

At runtime, the checkpoint contained `lm_head.weight`, but the active
`Qwen25_7BVLI` model instance did not expose a matching `lm_head` module.

This meant the checkpoint's generation head was not actually being used by the
local rewrite path.

### 2. `BaseGenerate.logits` effectively fell back to `embed_tokens.weight`

The generate code path patched in
`runtime_patches.py` (`_patch_qwen25_think_generation()`)
showed that logits were being formed from the embedding matrix instead of the
checkpoint `lm_head.weight`.

In practice, that meant AR generation was decoding against the wrong weight
matrix. The result was garbled continuation text even when tokenization,
prompting, and stop-token settings were otherwise reasonable.

### 3. Special-token drift was real, but secondary

`<|vision_end|>`-class special tokens were climbing into top logits during
generation. Suppressing them helped, but did not fully solve the problem.

That told us:

- special-token drift was a contributing factor
- the deeper issue was the logits / decode path itself

## Fix

The fix was applied inside the custom-node-owned runtime patch layer, not by
editing the reference ComfyUI checkout.

### 1. Add an `lm_head` sink for `Qwen25_7BVLI`

In `runtime_patches.py`,
`Qwen25_7BVLI.__init__` is patched so the model gets an `lm_head` module with
the correct `(vocab_size, hidden_size)` shape when one is not already present.

The patch also marks `config.lm_head = True` so the generate path treats the
model as head-backed.

### 2. Fix `BaseGenerate.logits` to use the active module weight

In `runtime_patches.py`,
`BaseGenerate.logits` is patched so it uses:

- `self.model.lm_head.weight` when `lm_head` exists
- `self.model.embed_tokens.weight` only as the fallback

This is the core behavioral fix.

### 3. Keep special-token suppression for rewrite-style generation

The rewrite-specific suppression path remains in place via
`nodes.py` (`_generate_with_rewrite_suppression()`)
so that known Qwen multimodal special tokens do not dominate the continuation.

This is still useful, but it is no longer the only thing standing between
thinking mode and total corruption.

## Evidence After the Fix

With the patched branch, debug logs for real `i2v` think runs now show coherent
English continuation text instead of multilingual garbage.

Representative pattern:

> "The video begins with a static, high-quality illustration ..."

That confirms the AR branch moved from "broken decode" to "semantically usable
but still quality-sensitive."

The remaining issue is no longer text corruption. It is prompt quality and
prompt drift, for example:

- adding too many inferred visual details
- over-describing clothing/background/object attributes
- introducing motion details not explicitly requested

## How This Now Affects `merge_hidden`

### Short answer

**It now matters directly.**

### Why

The current `merge_hidden` branch in `nodes.py`
(`think_mode == "merge_hidden"`) now does the following:

1. encode the original prompt as the base branch
2. run AR prompt enhancement through the same Qwen generate path
3. take the AR `generated_text`
4. encode that generated text as the auxiliary branch
5. merge hidden states with concat

So, unlike the earlier deterministic-suffix implementation, the current
`merge_hidden` path depends on `clip.generate(...)` being correct.

That means:

- the `lm_head` / logits fix is required for `legacy_rewrite`
- the same fix is also required for the current `merge_hidden`

## Indirect Consequences

Because both think paths now rely on the AR branch:

- local probe tooling for rewrite quality remains relevant
- suppression behavior matters for both modes
- if AR generation regresses again, both `legacy_rewrite` and `merge_hidden`
  will regress, just in different ways

Expected failure mode split:

- `legacy_rewrite`: broken or noisy rewritten prompt text
- `merge_hidden`: broken or noisy auxiliary hidden-state branch

## Practical Conclusion

- The garbled think output was primarily caused by the local Qwen generate path
  not using the checkpoint `lm_head` correctly.
- The fix was to patch Qwen init and logits selection inside the custom-node
  runtime patch layer.
- `legacy_rewrite` now appears functional enough to evaluate on quality.
- `merge_hidden` now also depends on this fix because it uses AR-generated
  text as its auxiliary branch input.
- The main open issue has shifted from decode correctness to branch content
  quality and visual stability.
