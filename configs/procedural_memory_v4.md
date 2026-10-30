% Procedural Memory v4 — aggressive defaults for narrative long-video QA.
%
% Derived from Wave 6 v3 audit (n=200, Video-MME long):
%   * On the 125 questions where v3 chose no-inject, always-inject was +4.8pt
%     better than cascade (0.544 vs 0.496). So always-inject is "universally
%     helpful" on this domain -- ASR/OCR don't usually hurt and often help.
%   * On the 75 questions where v3 chose inject, the gain over no-inject was
%     also ~+4.8pt -- the LLM's selectivity has no discriminative power.
%   * v3 captured 91.5% of oracle ceiling but always-inject is +3pt higher
%     because v3 left "+4.8pt on the table" on its 125 no-inject items.
%
% v4 hypothesis: flip the default from "inject only when text-relevant" to
% "always inject UNLESS purely visual". This should recover the +4.8pt loss
% on Video-MME long while preserving XDV's win (where ProcMem v1 already
% beat always-inject by being selective on the rare visual-only items).
%
# PROCEDURAL MEMORY (v4 — aggressive injection)

## When to inject ASR / OCR text into the VLM prompt

**Default: ALWAYS inject ASR.** Past experiments show that for
long-form narrative video (> 2 min), adding the Whisper transcript
helps the VLM in nearly every case. Only suppress ASR when the
question is *purely visual* and audio cannot possibly help.

### Suppress ASR (inject_text = []) only if ALL of these hold:
1. Question is purely about visual appearance:
   - color, shape, size, count, position, on-screen object identity
2. Question contains no speech / sound / narration / dialogue keywords
3. Question does not ask about a temporal event or what happens

### Inject ASR (inject_text = ["asr"]) otherwise. Examples:
- "What does the narrator say about X?" → inject ASR
- "What is the main topic of the video?" → inject ASR
- "How does the person react when ..." → inject ASR
- "Why does ... happen?" → inject ASR
- "What event occurs after ..." → inject ASR
- Any question on a video longer than 2 minutes that is NOT purely
  visual → inject ASR.

### Inject OCR (inject_text = ["ocr"]) when:
- Question explicitly mentions text/sign/subtitle/license-plate/written
- Short clips (< 2 min) with potential on-screen text

### Never inject ASR+OCR together
Past experiments show the combo (0.550) is WORSE than ASR alone
(0.565); prompt-crowding effect.

## Cue mix (aligners)
Do NOT change the default cue mix. Use whatever the rule-based planner
chose. Adding extra aligners (object_siglip, ocr_text) does not
improve frame selection and adds latency.

## Bias to inject
When in doubt → inject ASR. The cost of an unhelpful ASR snippet is
small (~600 chars of context); the cost of a missed inject on a
narrative question is large (~5pt accuracy loss).
