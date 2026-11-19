# SO-101 catch policy: deployment safety notes

Status: everything below is validated in simulation only. NOTHING has been run on real hardware.

## 1. Command path (policy -> arm)
`policy chunk -> SmoothPolicyController (fixed noise, chunk blending, EMA) -> SafetyFilter -> SO-101 follower`

SafetyFilter (`challenge/sim/safety.py`), in this order:
1. non-finite command -> hold the last safe command
2. joint range clamp: hardware range minus 6 deg (pan +-110, lift +-100, elbow +-96.8, wrist flex +-95, roll -157..163); gripper 0..100
3. per-step speed limit: 12 deg/step arm (360 deg/s), 35 units/step gripper
4. table check with the MuJoCo kinematic model: reject poses that push the arm more than 20 mm into the table (skimming contact is normal)
5. stall guard: measured joint more than 35 deg behind the command for 15 steps -> stop pushing, hold the measured pose

Also set in LeRobot: `SO101FollowerConfig(use_degrees=True, max_relative_target=10.0)` so the driver clamps the step relative to the
measured position as a second, independent limit.

Measured in simulation: with all layers on, the scripted expert's catch rate is identical to the unfiltered baseline on the same seeds
(31/40 easy, 27/40 hard under full randomization), because the expert's commands already stay inside the limits
(wrist flex never below -85 deg). The expert's peak joint speed is 15 deg/step on rare steps; the filter clamps those.

## 2. Pre-flight checklist (before any power-on with the policy)
- Recalibrate in degrees mode and compare the joint ranges with the table above. If a joint range is smaller, tighten `RANGE`.
- Camera placement must match the sim cameras (top overhead, front): the policy tolerates about +-2 cm and +-2.5 deg of jitter (trained range).
- Lighting: trained on 0.6x to 1.4x of the nominal light; avoid strong colour casts or the ball being the same colour as the table.
- Ball: red, about 4 cm diameter, rolls without much slip on a flat table. Different size or a bouncy/soft ball is untested.
- Start with the arm at HOME (0, -60, 60, 35, 0, 30) and the workspace clear except for the ball.
- First runs: no ball, policy outputs only logged (dry run), then arm moving with the ball absent, then slow ball.
- Keep a hand on the power switch / have the leader arm or e-stop within reach. Never leave the arm powered and unattended.

## 3. Stop conditions to implement in the deployment loop
- loss of camera frames or tracker for more than 0.3 s -> hold position, open nothing, do not lift
- SafetyFilter stall counter fires -> hold pose and require a manual restart
- any joint within 6 deg of its limit for more than 1 s -> hold
- grip force: the sim cannot tell how hard the real gripper squeezes; start with the gripper close target at a small positive value
  (not 0) and check the servo load before trusting full closure

## 4. Known limits of the simulation (what can still go wrong)
- Jaw/ball contact physics, ball compliance and the real servo torque limits are not modelled faithfully. The catch uses
  the gripper in an unusual pose (wrist roll about -90 deg, nearly flat on the table); check clearance by hand first.
- Servo speed limit was randomized at 7-12 deg/step; a real arm under load or at low battery may be slower.
- The tracker in the sim is Gaussian noise with dropouts; a real ArUco/YOLO tracker has different failure modes.
- TabPFN forecast delay was simulated at 0.15-0.30 s; on other hardware it may differ.
