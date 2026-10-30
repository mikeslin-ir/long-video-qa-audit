# Procedural memory v1 -- mined from Video-MME long-form QA execution logs
# (Wave 1-4, n=200, K=8, Qwen2.5-VL-7B; commit 9e3bf8b head).

Past empirical findings (HIGH PRIORITY -- override your default conservatism):

1. LONG documentary / educational videos (duration > 120s) almost always
   contain dense narration that carries the answer EVEN when the question
   does not literally mention a narrator. For questions like
   "What is the video mainly about?" / "What does the speaker explain?" /
   "Why does X happen?" / any "how" or "why" question on a long video,
   DEFAULT to inject_text=["asr"]. In our logs this gave +6.5 pt accuracy
   (p=0.066) versus no injection. Skipping injection (the conservative
   default) lost 4 pt.

2. For purely VISUAL attribute questions on long videos -- color, count,
   spatial position, what someone is wearing, who is holding what --
   ASR injection adds noise. Use inject_text=[] for those.

3. DO NOT inject both ASR and OCR simultaneously. In our logs the combo
   ASR+OCR lost 1.5 pt versus ASR alone (0.550 vs 0.565). The VLM prompt
   becomes too crowded and attention is diluted. Pick at most one text
   stream.

4. For on-screen-text questions ("What is written on the sign?",
   "What text appears on screen?", "What is the headline?",
   "What channel is broadcasting?"), use inject_text=["ocr"] only.

5. For SHORT clips (duration <= 120s) that are anomaly / event detection
   ("Does this video contain a fight? / theft? / accident?"), OCR of
   on-screen captions (chyrons, sirens, alarm text) sometimes helps;
   ASR is usually too sparse. Prefer inject_text=["ocr"] when in doubt.

Decision shortcut for long videos:
  - Visual-attribute question        -> inject_text=[]
  - On-screen text question          -> inject_text=["ocr"]
  - Otherwise (general / narrative)  -> inject_text=["asr"]   (this is the
                                                                 most common
                                                                 correct call)
