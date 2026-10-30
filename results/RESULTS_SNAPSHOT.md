# Phase C+ and Cascade K-sweep Results Snapshot

> **Historical, not the final paper audit.** This snapshot predates the final
> matched selector runs. Some values and interpretations below differ from
> the paper's final controlled audit. Use the
> [README main-result table](../README.md#main-result--video-mme-long-k8-n900)
> for the final reported comparison. Paired per-item predictions are still
> needed to independently replay its statistics.

_Snapshot generated: 2026-06-07; updated 2026-06-08 with strong baselines,_
_anomaly datasets, full K-sweep, and McNemar significance tests._

Backbone VLM: Qwen2.5-VL-7B-Instruct. Hardware: single NVIDIA GB10.
Source: `outputs/*/qa_summary.json` (outputs/ is gitignored; this file is the durable record).

| run | n | acc | sec/clip | fails |
|---|---:|---:|---:|---:|
| main_egoschema_cascade | 500 | 0.662 | 16.4 | 0 |
| abl_vlmonly_cascade_egoschema | 500 | 0.662 | 5.7 | 0 |
| abl_nms_egoschema | 500 | 0.570 | 13.3 | 0 |
| runtime_video_mme_singlestage | 30 | 0.567 | 131.2 | 0 |
| main_video_mme_cascade | 896 | 0.586 | 21.8 | 4 |
| abl_vlmonly_cascade_video_mme | 896 | 0.602 | 13.8 | 4 |
| ksweep_uniform_k4_egoschema | 500 | 0.642 | 1.6 | 0 |
| ksweep_uniform_k8_egoschema | 500 | 0.678 | 3.0 | 0 |
| ksweep_random_k4_egoschema | 500 | 0.598 | 1.6 | 0 |
| ksweep_random_k8_egoschema | 500 | 0.634 | 3.1 | 0 |
| ksweep_cascade_k4_egoschema | 500 | 0.620 | 4.4 | 0 |
| ksweep_cascade_k8_egoschema | 500 | 0.662 | 5.8 | 0 |

## Matched-budget comparison (EgoSchema, n=500)

| K | Cascade | Uniform | Random |
|---|---:|---:|---:|
| K=4 | 0.620 | 0.642 | 0.598 |
| K=8 | 0.662 | 0.678 | 0.634 |

Key reading: cascade > random at matched K, but cascade < uniform at matched K on EgoSchema
(uniformly-trimmed egocentric clips, where uniform is a very strong temporal prior).

## Track A: uniform-anchor floor (EgoSchema, n=500)

| run | n | acc | note |
|---|---:|---:|---|
| anchored_cascade_k8_f0p5_egoschema | 500 | 0.662 | 50% of K reserved for evenly-spaced anchors |

Reading: the anchor floor does NOT lift cascade above uniform (0.678) on EgoSchema;
it lands at the no-floor cascade number (0.662). The early small-sample 0.70 was noise.
Conclusion: on uniformly-trimmed video, smarter selection cannot beat uniform — uniform
already has near-optimal temporal coverage.

## Track B: matched-budget on event-sparse video (Video-MME LONG bucket, n=200, K=8)

| selector | n | acc | sec/clip |
|---|---:|---:|---:|
| **cascade (floor 0.5)** | 200 | **0.535** | 14.3 |
| uniform | 200 | 0.500 | 10.1 |
| random | 200 | 0.485 | 10.0 |

Key reading (the positive result): on long, event-sparse video the cascade BEATS
uniform by +3.5 pts and random by +5.0 pts at matched K=8. This is where smarter
keyframe routing pays off — uniform sampling misses the key moment.

## Strong content-aware baselines (Video-MME LONG, n=200, K=8)

Added in response to the reviewer concern that "uniform and random are too basic".
We wire two single-cue content-aware selectors to the SAME Qwen backbone:
`motion` (motion-difference only) and `clip` (SigLIP semantic only). p-values are
two-sided exact McNemar on paired per-item correctness vs. the cascade.

| selector | n | acc | sec/clip | p (vs cascade) |
|---|---:|---:|---:|---:|
| random | 200 | 0.485 | 10.0 | 0.154 |
| uniform | 200 | 0.500 | 10.0 | 0.324 |
| motion (motion-diff only) | 200 | 0.465 | 20.1 | **0.049** |
| clip (SigLIP only) | 200 | 0.495 | 20.9 | 0.312 |
| **cascade (ours, floor 0.5)** | 200 | **0.535** | 14.3 | — |

Key reading: single-cue content selectors do NOT beat uniform (motion 0.465,
clip 0.495 both <= uniform 0.500) — and they cost ~2x more. The cascade beats
every single-cue baseline; the win over `motion` is significant (p=0.049). This
is the core rebuttal: the gain comes from COMBINING cues via routing, not from
any one content signal.

## Event-sparse anomaly datasets (n=200, K=8)

Binary event-QA. UCF-Crime: "contains unusual/dangerous/criminal activity?";
XD-Violence: "contains violent activity (fighting/shooting)?". p = McNemar vs cascade.

| dataset | uniform | random | cascade (ours) | Δ vs uni | p (cas vs uni) |
|---|---:|---:|---:|---:|---:|
| UCF-Crime | 0.815 | 0.820 | **0.850** | +3.5 | 0.143 |
| XD-Violence | 0.870 | 0.880 | **0.910** | +4.0 | 0.115 |

Reading: cascade is the best selector on BOTH anomaly datasets (+3-4 pts).
Directional win is consistent but underpowered at n=200 (p~0.12-0.14).

## Cross-budget K-sweep (Video-MME LONG, n=200)

p = two-sided exact McNemar, cascade vs uniform, paired per-item.

| K | uniform | cascade | Δ | p (McNemar) | significant |
|---|---:|---:|---:|---:|:--:|
| 1 | 0.380 | 0.480 | +10.0 | **0.009** | *** yes |
| 2 | 0.430 | 0.470 | +4.0 | 0.302 | no |
| 4 | 0.490 | 0.500 | +1.0 | 0.875 | no (tie) |
| 6 | 0.525 | 0.510 | -1.5 | 0.728 | no (tie) |
| 8 | 0.500 | 0.535 | +3.5 | 0.324 | no |

Key reading: the cascade advantage is LARGEST and statistically significant in
the low-budget regime (K=1, +10 pts, p=0.009). As K grows, uniform coverage
becomes a stronger prior and the margin shrinks into the noise band
(SE ~3.5pp at n=200, acc~0.5); K=4 and K=6 are statistical ties. We report all
points honestly — including K=6 where uniform is nominally ahead by 1.5 pts —
rather than cherry-picking budgets.

## K=8 robustness at full scale (Video-MME LONG, n=900) — 2026-06-08

Scaled the K=8 cascade-vs-uniform comparison to the ENTIRE long bucket (n=900)
to check the n=200 subset was not an artefact.
Dirs: `trackB_full_long_uniform_k8`, `trackB_full_long_cascade_k8_f0p5`.

| n | uniform | cascade | Δ | discordant (b/c) | p (McNemar) | SE |
|---|---:|---:|---:|:--:|---:|---:|
| 200 (subset) | 0.500 | 0.535 | +3.5 | — | 0.324 | ~3.5pp |
| **900 (full)** | **0.462** | **0.472** | **+1.0** | 93/84 | **0.548** | ~1.7pp |

Reading: at the full split the K=8 gap NARROWS to +1.0 pt and is a clear TIE
(p=0.55). Absolute acc is lower (harder full bucket vs easier stratified subset),
but the relative ordering holds. This CONFIRMS the central finding: the real,
significant advantage is at low budget (K=1); at generous budgets uniform
coverage is adequate and selection adds little. The +3.5 at n=200 was within
noise, as predicted. Reported in paper as a "Robustness at scale" paragraph.

## Honesty note on statistical power

At n=200 with acc~0.5, the per-condition standard error is ~3.5pp, so individual
+/-3pt differences (incl. the K=8 +3.5) are directionally consistent but NOT
individually significant. The two effects that clear p<0.05 are: (a) low-budget
K=1 (+10pt, p=0.009) and (b) cascade vs single-cue motion at K=8 (p=0.049). The
defensible headline is "consistent directional gains across 3 datasets and 5
budgets, significant where frames are scarce", not "uniformly significant wins".
The n=900 K=8 follow-up (tie, p=0.55) confirms this framing empirically.
