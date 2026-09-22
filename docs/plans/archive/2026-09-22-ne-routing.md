# NE Routing Investigation — Experiment Log (2026-09-22)

Status: MEASURED (probes only, no shipped code), **routing designs mostly
FALSIFIED; generation-side work indicated**. Probe removed. Test splits used
for analysis only (established practice, same as eval-errors).

## Structure of NE misses (test AK-NEI n=1176, AK-NEF n=817)

| | NEI | NEF |
|---|---|---|
| top-1 hits | 535 (45.5%) | 238 (29.1%) |
| buried, gold in 2..10 | 352 (52% of misses) | 255 (43%) |
| — of which freq-buried (freq(top1) > freq(gold)) | 101 | 67 |
| deep, gold in 10..50 | 37 | 41 |
| absent from top-50 (true generation gap) | 265 (41%) | 294 (50%) |
| gold OOV among misses | 346 | 298 |
| hits with OOV top-1 (novel names) | 91 | 47 |
| demotable (OOV top1 + in-vocab gold in top-10) | 70 | 70 |

Qualitative: entity errors concentrate in nasal placement (वेलिंगटन vs
वेलिङ्टन), vowel length (स्कटिस vs स्कॉटिश), schwa+nasal (सान्ता vs सन्त)
and retroflex choice (टिनले vs तिनले) — channel-level confusions on
foreign phonetics, not ranking noise.

## Designs killed

1. **Wider beam for entity-looking inputs (router).** Deep mass is only
   78/1220 (6%). Worse: requesting depth-200 vs depth-50 CHANGED top-1 by
   −24 net on these sets — deeper decode adds distractors the weak ranker
   mis-orders (same mechanism as the beam-128 ship-gate rejection). A router
   feeding more candidates to this ranker is contraindicated by measurement.
2. **OOV-top-1 demotion.** Ceiling +140 (70+70) vs breakage −138 (91+47
   novel-name hits it would destroy) — no safe operating point without a
   channel-score gate, which is itself untried machinery for ~+50 best case.
   Parked, not pursued.
3. **Prior attenuation as primary lever.** Freq-buried is 168/1220 (14%) and
   every reweighting experiment on record harms native. At most a small
   entity-gated heuristic term (runtime-only, no retrain) — optional polish,
   not a lever.

## Indicated: generation-side (needs a training run)

559/1220 misses (46%) have gold absent from the top-50: no rerank rule can
touch them. The errors are foreign-phonetic emissions (retroflex,
aspirates, o-length, nukta consonants, nasal placement) drowned by
native-dominant EM counts. Structural fix: attestation-weighted training
(`sota-signals.md` T4 — weight pairs by variant frequency so convention
dominance is learned) and/or entity-upsampled EM, validated by
`train-mid` + entity-stratum deltas. That is a ~40-min machine-time bet,
not a code-design bet.
