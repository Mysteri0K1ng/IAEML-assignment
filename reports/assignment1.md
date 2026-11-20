# Assignment 1: vehicle environment

**Author:** Nikita Shankin  
**Typed signature:** Nikita Shankin  
**Date:** November 20, 2026  
**Status:** Includes the author's personal reflection.

## Implementation

`src/env/robotaxi.py` defines `RoboTaxiEnv(BaseEnv)`. The original `base.py` is unchanged. The new environment supplies the same initialization, reset, observation, and five-value step interface. State and observation objects are Equinox pytrees; the numerical simulation uses JAX throughout. Both `jax.jit` and `jax.vmap` are supported.

The world uses metres and seconds, with x pointing right and y pointing up. Every vehicle and obstacle has collision radius 1 m. Goal arrival means that the vehicle centre enters the goal's 1 m acceptance disk. The goal is a destination, not a collision obstacle.

## State and actions

The state includes position `(x,y)`, unit heading `(cos(psi),sin(psi))`, forward speed `v`, front-wheel steering `delta`, step count, goal, static circles, moving-circle positions and velocities, and a fixed-size route. Terminal flags record success, collision, and time-limit truncation separately.

The continuous action is `(u_a,u_delta)` in `[-1,1]^2`. The first component commands acceleration or braking; the second commands the desired steering angle. The mapping is `a_command=3*u_a` m/s² and `delta_command=0.6*u_delta` rad. Braking reduces speed to zero; reverse driving is excluded. Steering changes at most 1.5 rad/s. Sanitization and clipping keep invalid external actions from propagating NaNs.

## Dynamics model and justification

The implementation uses a **kinematic bicycle model with a longitudinal drag model**. It does not implement lateral tyre-force dynamics. This is a reasonable first model for the low-speed task (maximum 4 m/s), because it provides nonholonomic steering without introducing tyre parameters that the assignment does not specify.

The wheelbase is `L=1.6 m` and the centre-of-mass rear fraction is `l_r/L=0.5`. With body heading `psi`:

```text
delta_next = delta + clip(delta_command - delta, -1.5*dt, 1.5*dt)
a = 3*u_a - 0.15*v - 0.04*v^2
v_next = clip(v + dt*a, 0, 4)
beta = atan(0.5*tan(delta_next))
x_next = x + dt*v_next*cos(psi + beta)
y_next = y + dt*v_next*sin(psi + beta)
psi_next = psi + dt*v_next*cos(beta)*tan(delta_next)/L
```

The drag terms represent effective rolling/drivetrain resistance and speed-dependent aerodynamic resistance after dividing force by vehicle mass. They dissipate forward motion when the throttle is released. They are simplified acceleration coefficients, not fitted physical constants. There is no road slope, tyre saturation, or lateral sliding. The algebraic slip angle `beta` follows the bicycle geometry and should not be confused with a dynamic tyre-slip state.

An environment action lasts 0.1 s and is integrated in four 0.025 s substeps. Speed is updated before position. This keeps the implementation inexpensive and makes the numerical approximation explicit. Vehicle heading is reconstructed as a unit vector after each substep.

Moving obstacles have constant velocity between reflections from the rectangular boundary. Their reflection respects their 1 m radius. They do not avoid one another or static obstacles; they are kinematic hazards rather than independently controlled cars.

## Collisions and episode endings

Two radius-one objects collide when the centre separation is at most 2 m. This corrects the starter's radius-one centre-distance test in the subclass. Each substep checks minimum separation along the relative-motion segment, which catches a moving obstacle crossing the car between endpoints. Boundary contact also ends the episode. The vehicle centre must remain one radius inside the world rectangle.

The continuous test treats each integration substep as a straight segment; a finer integration step would improve the approximation to a curved bicycle trajectory. Contact has priority over goal arrival within a substep. A successful arrival or collision is a true terminal transition; reaching 240 actions is a truncation. Finished states are absorbing and return zero reward until an explicit reset.

## Observation

The policy receives 24 float32 values:

| Component | Size | Scaling |
| --- | ---: | --- |
| Unit direction to goal in vehicle coordinates | 2 | Components in `[-1,1]` |
| Goal distance | 1 | Divided by map diagonal |
| Forward speed | 1 | Divided by 4 m/s |
| Actual steering angle | 1 | Divided by 0.6 rad |
| Direction to route lookahead in vehicle coordinates | 2 | Unit vector |
| Distance to nearest route waypoint | 1 | Divided by map diagonal |
| 360-degree ray distances | 16 | Divided by 8 m range |

Rays measure distance from the vehicle centre to obstacle surfaces and map boundaries. The car's own body radius is not subtracted from the ray values. Ray zero points backwards; subsequent rays sweep counterclockwise, with the forward ray at index 8. The reward and collision checks use geometric clearance rather than treating ray length as centre distance.

The observation exposes neither obstacle velocity nor full global geometry. It is therefore partially observed for moving hazards. A recurrent policy or stacked observations would be a reasonable extension. The route comes from the environment's privileged static map, following the starter's navigation assumption.

## Navigation and maps

The subclass uses a bounded eight-neighbour dynamic-programming planner. Static circles are inflated to 2.15 m to allow for the vehicle radius and a small planning margin. Padded logical boundaries prevent wraparound, and diagonal corner cutting is disallowed. Bellman relaxation and fixed-length path tracing execute in JAX. Moving hazards are handled by the policy's perception and are not frozen into the static route.

The route is computed once for the template start and destination. Randomized resets share this route and join it using the nearest-waypoint lookahead. The planner is not rerun at every simulation step. Unreachable trace positions remain in place, so the route is a hint rather than a feasibility certificate for the vehicle's turning radius.

Map 3 is a new 24 by 20 m training arena. Its nominal start is `(4,10)`, goal `(20,10)`, static obstacles `(10,6)`, `(13,14)`, `(18,5)`, and moving obstacle `(14,16)` with velocity `(0,-0.6)` m/s. Reset randomization changes start x to `[3,5]`, start y to `[7,13]`, heading by up to 0.7 rad from the goal direction, and moving-obstacle y to `[4,16]`.

Original maps 0-2 remain selectable. They use an explicit default spatial scale of 3 for obstacle centres, start, goal, and map extent; circle radius remains 1 m and obstacle velocities remain in m/s. This changes physical spacing, so results on map 3 or scaled maps must not be presented as results on the original unit-spaced geometry. The scale is configurable through `init_params(world_scale=...)`.

## Reward and justification

For an active transition, let `d` and `d_next` be Euclidean goal distance and let `c` be the smallest surface clearance to any obstacle or boundary. The reward is:

```text
r = 2*(d - d_next)
    - 0.01
    - 0.002*(u_a^2 + u_delta^2)
    - 0.02*max(0, 1-c)
    + 25*success
    - 15*collision
```

Progress supplies a dense learning signal before the first successful episode. The time cost discourages waiting, the small control penalty discourages extreme controls, and clearance cost discourages near misses. The terminal rewards distinguish actual completion from temporary progress. An absorbing state receives zero rather than repeated terminal rewards.

This is a task-specific shaped reward, not an assertion of policy-invariant potential shaping for discounted RL. Euclidean progress can discourage a necessary detour in difficult mazes. The goal bonus can also reward entering the destination while still moving; precise stopping is outside this experiment's objective.

## Validation, video, and reproduction

The tests check acceleration, drag, heading, steering-rate limits, radius-sum collisions, swept collisions, terminal and truncation flags, absorbing states, deterministic randomized resets, sensor distances, empty obstacle arrays, JIT/vectorization, and planner bounds and clearance. Run the checks and recorder using the setup in the repository README.

```bash
python -m pytest -q
python -m demo --policy heuristic --seed 0 --fixed-start --output results/environment.mp4
python -m demo --interactive
```

The reference-controller video is [environment.mp4](../results/environment.mp4); its machine-readable outcome is [environment.json](../results/environment.json). The controller uses geometric waypoint pursuit and is separate from the RL policy. All displayed circle radii match collision geometry. W/S control throttle and brake, A/D control steering, and R resets in the interactive frontend.

## Engineering observations

The implementation addresses three concrete issues in the starter: a radius-one collision threshold for two circles, a navigation relaxation that rolls across map edges, and episode status being checked before motion. The subclass uses radius-sum geometry, bounded navigation, and checks the resulting transition. Small physics substeps and a swept relative-motion test address missed collisions.

## Personal Reflection 1

The main thing I needed to understand in this assignment was how the different parts of the environment work together. It is easier for me to understand something when I separate it into smaller steps instead of just reading the theory.

In this environment, the action controls acceleration and steering, the bicycle model updates the vehicle position, and then the environment checks collisions and calculates the reward. The kinematic bicycle model is a reasonable choice because we do not need fully realistic vehicle physics for this task.

One important technical problem in the implementation was collision detection. Since both the vehicle and obstacles have radius 1, a collision happens when the distance between their centers is 2 or less. The implementation also checks movement between simulation steps, which helps prevent missed collisions. These cases are covered by tests.

The environment is still simplified. For example, it does not simulate all real vehicle forces, and the agent does not directly observe moving-obstacle velocities. If I continued this assignment, I would try more difficult maps and moving obstacles to see how the environment and navigation behave in different situations.

## Sources

- Starter code and map definitions: [InnoRL/IAEML-assignment](https://github.com/InnoRL/IAEML-assignment), upstream commit `1056d71` (full identifier in `results/source.json`).
- Bicycle-model background linked by the assignment: [course video](https://www.youtube.com/watch?v=wuUUN_DvYP4&list=PL05umP7R6ij321zzKXK6XCQXAaaYjQbzr&index=16). The implementation is an explicit kinematic simplification; no code was copied from the lecture.
- [JAX documentation on JIT compilation](https://docs.jax.dev/en/latest/jit-compilation.html) for the transformation and compilation model.
