# What has been tried, in order

A plain account including the things that did not work. Numbers refer to [RESULTS.md](RESULTS.md).

## Phase A: from LIBERO to the SO-101
- Started with SmolVLA on LIBERO (Franka Panda): local dataset, fine-tuning trials, a closed-loop rollout in a WSL simulator
  ([LIBERO_LEGACY.md](LIBERO_LEGACY.md)). The LIBERO checkpoint cannot drive an SO-101 (different action space and statistics), so the
  challenge work restarted on the SO-101.
- **Stage 1**: fine-tuned `smolvla_base` on real SO-101 pick-and-place data (`lerobot/svla_so101_pickplace`) to learn the
  robot's action space and cameras.

## Phase B: block the rolling ball (superseded)
- **Simulator**: MuJoCo SO-101 tabletop, red ball, launch lane, blocking line, paddle on the gripper; hidden dynamics (rough patch,
  slope, nudges) in the "hard" split.
- **TabPFN forecaster**: two TabPFN v2 regressors on residuals against a constant-velocity baseline; forecast every 3 frames; compute
  delay measured at about 0.2 s. A correction filter over successive forecasts was tested and **hurt**, so it is off.
- **Stage 2**: 15,000 steps on exact-target demonstrations: the policy used the target (blocked 50/50 with the exact one, 28/50 with none).
- **Stage 3 (negative result)**: re-trained on demonstrations whose targets were the noisy TabPFN forecasts, hoping to teach it to
  cope with noise. It ignored the target altogether. The lesson drove the mixed-target data later.

## Phase C: catch and hold (the current task)
Direction change from the user: catch and hold the ball instead of blocking it, keep the ball speeds (0.10–0.28 m/s), and make
smoothness count in the score.
1. **A catch expert from scratch.** First attempts (gripper pointing at the ball, jaws opening vertically) caught almost nothing.
   What worked: the mouth faces the ball, the wrist is rolled to about −90° so the jaws open sideways, pitch 12°, jaws fully
   open (smaller openings cost many catches), a small sideways aim correction, closing 0.25 s early, a 0.9 s lift. A fixed catch plane
   (x = 0.24, the forecast plane) cost a few points against per-launch planes but keeps the demonstrations unambiguous.
   An earlier "99% easy" figure was with per-launch planes; with the fixed plane the expert holds about 89% easy and 78–82% hard in
   the clean simulator, about 83% and 77% under full randomization.
2. **Odd-looking pose.** The user asked why the gripper looks abnormal: the gripper lies almost flat on the table with the camera
   on top. That is a consequence of the geometry (side-opening jaws). Joint margins of 10° from the hardware limits and a
   table-penetration check were added so the pose is not at the edge of the joint ranges.
3. **Jitter.** The raw policy is very jerky. A smoothing controller (fixed flow-matching noise, chunk blending, EMA) cut
   jerk 4.4× on the blocking policy and 5–6× on the catch pilot. Jerk is measured as the RMS third difference of the joint commands.
4. **Sim-to-real layer** ([REPRODUCE.md](REPRODUCE.md#5-sim-to-real-layer)) and the safety filter
   ([DEPLOYMENT_SAFETY.md](../DEPLOYMENT_SAFETY.md)). A first table-collision rule at 12 mm intervened too often and cost many
   catches; 20 mm kept the expert's catch rate unchanged.
5. **Data**: 400 demonstrations with the mixed-target recipe, 25% clean and 75% randomized.
6. **Pilot first** (3,000 steps, about 35 minutes), then the big run only after the pilot showed that the policy catches.
7. **Evaluation** with a fair launch set (the scripted expert must be able to catch it), clean vs randomized, three target modes.

## Phase D: the checks that followed the big run
- **Is TabPFN forecasting properly?** Yes: accurate late, better than constant velocity on easy launches; the hard-launch gap is
  mostly unpredictable nudges (likely, not proven).
- **Time gate on the policy input** (target hidden until 0.6 s before arrival): much worse (2/24 vs 9/24). Discarded.
- **Marker removed** (real-world setting): fine with exact targets; with TabPFN targets the hard split collapses (1/24).
- **HSV-only tracking** of the ball for TabPFN: accurate in simulation (0.05 cm clean, 2 cm randomized); closed-loop results in
  [RESULTS.md](RESULTS.md#4-real-world-inputs-no-marker-hsv-only-tracking).
- **Spatial brain** (the second task): zone classification, confidence gate, few-shot adaptation. The LIBERO part and the
  multi-zone SO-101 demo were not done (time).

## Things deliberately not claimed
- Anything about a real arm, real contact physics, real servos, or a real tracker.
- That the scores are close to the expert's: the policy's smoothness is 2–4× worse and its catch rate under randomization
  and with TabPFN targets is well below the expert's.
- That one run per stage and 24 launches per cell separate small differences.
