# To do

Ordered by what most changes the outcome. Each item says what to test and how we would know it worked.

## Highest priority
1. **Retrain without the blue marker.** The policy leans on the dot when the forecast is noisy (hard launches with TabPFN
   targets: 9/24 with the dot, 1/24 without). Re-record the 400 demonstrations with the dot hidden (or never rendered), keep the 6
   numeric forecast values in the state, train from the current checkpoint, and re-run the marker-off / HSV evaluation. Needs
   about 6 h of recording and 2.5 h of training. Success: TabPFN targets on hard launches back to 9/24 or better without the dot.
2. **Bring jerk down to the score threshold (4000 deg/s³; now 9k–16k).** Ideas, cheapest first: stronger blending of chunks;
   executing fewer steps of each chunk; a jerk penalty or smoothed-label fine-tune (train on the expert's min-jerk trajectories
   resampled to the same chunk length); longer training. Success: oracle-target points per launch above 0.6.
3. **Train the policy to use the forecast when it is noisy**, not only exact: noise-augmented targets inside the training loop, and
   the forecast delay randomized (0.15–0.30 s) so late kick information is expected. Success: hard-launch TabPFN catch rate
   closes part of the gap to oracle (9/24 vs 19/24).

## The gate idea, done properly
4. Use the TabPFN uncertainty (sigma) to decide *when to commit the close and lift*, not to hide the forecast from the policy
   (hiding it from the input collapsed the result: 2/24). Test with the scripted expert first (cheap), then with the policy.
5. Hybrid executor: use the learned policy unless the gate reports low confidence, then fall back to the scripted catch expert.
   Needs the expert to start from an arbitrary arm pose (it currently plans from a pose near home).

## Queued task: TabPFN spatial brain + SmolVLA executor
6. **LIBERO backend** for benchmark numbers: pull a pre-finetuned SmolVLA LIBERO checkpoint (check that the repo ids exist
   before relying on them: `BurgQ/smolvla-libero-bs32-100k`, `crislmfroes/smolvla_libero`). LIBERO checkpoints do not
   transfer to the SO-101 (7-D end-effector deltas vs 6-D absolute joints, different normalization); never reuse their stats.
7. **SO-101 zones demo** (sim): 4–6 zones, 40–60 scripted demonstrations (10 per zone), canonical prompts per zone, fine-tune from
   `smolvla_base` with the new dataset's own statistics, clamp the step relative to the measured position when deploying.
8. **Live few-shot adaptation**: append 5–8 new examples to the TabPFN context at run time (no retraining of either model).
   The simulated adaptation curve exists ([RESULTS.md](RESULTS.md#5-tabpfn-spatial-brain-zone-prediction-uncertainty-gate-few-shot-adaptation));
   the same test is still to be run on the ball forecaster with a changed friction.
9. **Deterministic fallback executor**: four recorded trajectories, one per zone, behind a flag.
10. Answer the question first: does SmolVLA beat a plain IK controller on continuous (x, y) targets? The scripted expert holds
    about 80–90% of all launches (see EXPERIMENT_LOG), so the learned policy has to earn its place; use SmolVLA for the language-conditioned
    hierarchy, not for precision.

## Evidence we still lack
11. More launches per cell (n ≥ 60) and several training seeds to put error bars on the comparisons.
12. A run where the nudge time is known to the forecaster, to confirm that unpredictable nudges explain the oracle/TabPFN gap.
13. YOLO versus HSV on real frames (not simulated): lighting, glare, partial occlusion by the gripper.
14. **Real hardware.** Everything in [DEPLOYMENT_SAFETY.md](../DEPLOYMENT_SAFETY.md): recalibrate in degrees mode, dry run with
    policy outputs only logged, then the arm with no ball, then a slow ball, with a hand on the power switch.
15. Servo realism: a real STS3215 under load and at low battery can be slower than the 7–12°/step modelled; contact
    physics of the jaws and a real (softer or bouncier) ball are not modelled.
