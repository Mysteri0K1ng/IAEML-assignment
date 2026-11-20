"""Pure JAX bicycle dynamics. Original BaseEnv is deliberately left intact.

World coordinates are (x, y), angles are counterclockwise, all circles have
radius one. Actions in [-1, 1]^2 mean throttle/brake and desired front steering.
"""
from dataclasses import dataclass
from functools import partial
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from env.base import BaseEnv, BaseEnvState
from env.tasks import BaseTaskSampler
from utils.utils import convert_to_world_view


@dataclass(frozen=True)
class RoboTaxiParams:
    map_height_width: tuple = (20, 24)
    max_steps_in_episode: int = 240
    fps: int = 10
    substeps: int = 4
    wheelbase: float = 1.6
    rear_fraction: float = 0.5
    max_speed: float = 4.0
    max_acceleration: float = 3.0
    max_steer: float = 0.6
    steer_rate: float = 1.5
    rolling_drag: float = 0.15
    quadratic_drag: float = 0.04
    num_ray_sensors: int = 16
    perception_radius: float = 8.0
    agent_FOV: float = float(np.pi)
    path_length: int = 100
    randomize_reset: bool = True

    @property
    def step_size(self):
        return 1.0 / self.fps


class RoboTaxiState(BaseEnvState):
    speed: jax.Array
    steering: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    success: jax.Array
    collision: jax.Array


class RoboTaxiObservation(eqx.Module):
    features: jax.Array
    distance_to_path: jax.Array
    direction_of_path: jax.Array
    collision_rays: jax.Array

    def as_vector(self):
        return self.features


def _replace(state, **changes):
    return eqx.tree_at(lambda s: tuple(getattr(s, n) for n in changes),
                       state, tuple(changes.values()))


class RoboTaxiEnv(BaseEnv):
    @staticmethod
    def init_params(key, map_id=3, max_steps=240, path_length=100,
                    discretization_scale=1, perception_radius=8.,
                    num_ray_sensors=16, fov=float(np.pi), fps=10,
                    world_scale=3., randomize_reset=True):
        if discretization_scale != 1:
            raise ValueError("Only unit navigation-grid spacing is supported")
        if map_id not in range(4) or world_scale < 1 or fps <= 0 or max_steps < 1 or path_length < 2 or num_ray_sensors < 1:
            raise ValueError("Invalid map or environment dimensions")
        if map_id == 3:
            shape = (20, 24)
            start, goal = jnp.array([4., 10.]), jnp.array([20., 10.])
            static = jnp.array([[10., 6.], [13., 14.], [18., 5.]])
            moving, velocity = jnp.array([[14., 16.]]), jnp.array([[0., -0.6]])
            forward = jnp.array([1., 0.])
        else:
            task = BaseTaskSampler(map_id)
            shape = (int((task.height - 1) * world_scale + 1),
                     int((task.width - 1) * world_scale + 1))
            convert = lambda x: convert_to_world_view(x, (task.height, task.width)).astype(jnp.float32) * world_scale
            start, goal = convert(task.agent_pos), convert(task.goal_pos)
            static = jax.vmap(convert)(task.static_obstacles)
            moving = jax.vmap(convert)(task.kinematic_obstacles)
            velocity = task.kinematic_obst_velocities.astype(jnp.float32)
            forward = task.agent_forward_dir
            randomize_reset = False
        params = RoboTaxiParams(map_height_width=shape, max_steps_in_episode=max_steps,
                               fps=fps, perception_radius=perception_radius,
                               num_ray_sensors=num_ray_sensors, agent_FOV=fov,
                               path_length=path_length, randomize_reset=randomize_reset)
        path = RoboTaxiEnv._plan(start, goal, static, params)
        state = RoboTaxiState(time=jnp.array(0, jnp.int32), goal_pos=goal,
                             agent_pos=start, agent_forward_dir=forward,
                             static_obstacles=static, kinematic_obstacles=moving,
                             kinematic_obst_velocities=velocity, path_array=path,
                             speed=jnp.array(0.), steering=jnp.array(0.),
                             terminated=jnp.array(False), truncated=jnp.array(False),
                             success=jnp.array(False), collision=jnp.array(False))
        return params, state

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def reset(key, env_params, init_state):
        if init_state is None:
            raise ValueError("Supply the template returned by init_params")
        state = init_state
        if env_params.randomize_reset:
            k1, k2, k3 = jax.random.split(key, 3)
            pos = jnp.array([4., 10.]) + jax.random.uniform(k1, (2,), minval=jnp.array([-1., -3.]), maxval=jnp.array([1., 3.]))
            delta = state.goal_pos - pos
            yaw = jnp.arctan2(delta[1], delta[0]) + jax.random.uniform(k2, (), minval=-0.7, maxval=0.7)
            moving = state.kinematic_obstacles.at[0, 1].set(jax.random.uniform(k3, (), minval=4., maxval=16.))
            state = _replace(state, agent_pos=pos, agent_forward_dir=jnp.array([jnp.cos(yaw), jnp.sin(yaw)]),
                             kinematic_obstacles=moving)
        state = _replace(state, time=jnp.array(0, jnp.int32), speed=jnp.array(0.), steering=jnp.array(0.),
                         terminated=jnp.array(False), truncated=jnp.array(False),
                         success=jnp.array(False), collision=jnp.array(False))
        return RoboTaxiEnv.get_observation(state, env_params), state

    @staticmethod
    def _clearance(position, static, moving, params):
        obstacles = jnp.concatenate([static, moving], axis=0)
        gap = jnp.min(jnp.linalg.norm(obstacles - position, axis=-1), initial=jnp.inf) - 2.
        bounds = jnp.array([params.map_height_width[1] - 1., params.map_height_width[0] - 1.])
        wall_gap = jnp.min(jnp.concatenate([position - 1., bounds - 1. - position]))
        return jnp.minimum(gap, wall_gap)

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def step(key, env_state, action, env_params):
        p, s = env_params, env_state
        # Already-finished episodes are absorbing; reset is explicit.
        was_done = s.terminated | s.truncated
        action = jnp.clip(jnp.nan_to_num(action), -1., 1.)
        dt = p.step_size / p.substeps
        bounds = jnp.array([p.map_height_width[1] - 1., p.map_height_width[0] - 1.])

        def integrate(_, carry):
            pos, forward, speed, steer, moving, velocity, collision, success = carry
            active = ~(collision | success)
            steer_new = steer + jnp.clip(action[1] * p.max_steer - steer, -p.steer_rate * dt, p.steer_rate * dt)
            force = action[0] * p.max_acceleration - p.rolling_drag * speed - p.quadratic_drag * speed * speed
            speed_new = jnp.clip(speed + dt * force, 0., p.max_speed)
            beta = jnp.arctan(p.rear_fraction * jnp.tan(steer_new))
            yaw = jnp.arctan2(forward[1], forward[0])
            yaw_new = yaw + dt * speed_new * jnp.cos(beta) * jnp.tan(steer_new) / p.wheelbase
            pos_new = pos + dt * speed_new * jnp.array([jnp.cos(yaw + beta), jnp.sin(yaw + beta)])
            raw = moving + dt * velocity
            reflect = (raw < 1.) | (raw > bounds - 1.)
            moving_new = jnp.where(raw < 1., 2. - raw, jnp.where(raw > bounds - 1., 2. * (bounds - 1.) - raw, raw))
            velocity_new = jnp.where(reflect, -velocity, velocity)
            # Closest approach along relative motion catches tunnelling in a substep.
            old_obs = jnp.concatenate([s.static_obstacles, moving], axis=0)
            new_obs = jnp.concatenate([s.static_obstacles, moving_new], axis=0)
            rel0, rel1 = pos - old_obs, pos_new - new_obs
            dr = rel1 - rel0
            t = jnp.clip(-jnp.sum(rel0 * dr, axis=-1) / (jnp.sum(dr * dr, axis=-1) + 1e-8), 0., 1.)
            hit = jnp.any(jnp.linalg.norm(rel0 + t[:, None] * dr, axis=-1) <= 2.)
            hit |= RoboTaxiEnv._clearance(pos_new, s.static_obstacles, moving_new, p) <= 0.
            # Goal crossing is also continuous across the short integration segment.
            segment = pos_new - pos
            tg = jnp.clip(jnp.dot(s.goal_pos - pos, segment) / (jnp.dot(segment, segment) + 1e-8), 0., 1.)
            reached = jnp.linalg.norm(pos + tg * segment - s.goal_pos) <= 1.
            proposed = (pos_new, jnp.array([jnp.cos(yaw_new), jnp.sin(yaw_new)]), speed_new,
                        steer_new, moving_new, velocity_new, collision | hit, success | (reached & ~hit))
            return jax.tree.map(lambda new, old: jnp.where(active, new, old), proposed, carry)

        carry = (s.agent_pos, s.agent_forward_dir, s.speed, s.steering,
                 s.kinematic_obstacles, s.kinematic_obst_velocities, s.collision, s.success)
        pos, forward, speed, steer, moving, velocity, collision, success = jax.lax.fori_loop(0, p.substeps, integrate, carry)
        terminated = collision | success
        truncated = (s.time + 1 >= p.max_steps_in_episode) & ~terminated
        new = _replace(s, time=s.time + 1, agent_pos=pos, agent_forward_dir=forward,
                       speed=speed, steering=steer, kinematic_obstacles=moving,
                       kinematic_obst_velocities=velocity, terminated=terminated,
                       truncated=truncated, success=success, collision=collision)
        new = jax.tree.map(lambda old, x: jnp.where(was_done, old, x), s, new)
        progress = jnp.linalg.norm(s.agent_pos - s.goal_pos) - jnp.linalg.norm(new.agent_pos - s.goal_pos)
        clearance = RoboTaxiEnv._clearance(new.agent_pos, new.static_obstacles, new.kinematic_obstacles, p)
        reward = 2. * progress - 0.01 - 0.002 * jnp.sum(action**2) - 0.02 * jnp.maximum(0., 1. - clearance)
        reward += 25. * success - 15. * collision
        reward = jnp.where(was_done, 0., reward)
        obs = RoboTaxiEnv.get_observation(new, p)
        info = {"terminated": new.terminated, "truncated": new.truncated,
                "success": new.success, "collision": new.collision,
                "distance_to_goal": jnp.linalg.norm(new.agent_pos - new.goal_pos),
                "clearance": clearance, "time": new.time,
                "distance_to_path": obs.distance_to_path, "direction_of_path": obs.direction_of_path}
        return obs, new, reward, new.terminated | new.truncated, info

    @staticmethod
    def get_observation(env_state, env_params):
        s, p = env_state, env_params
        forward = s.agent_forward_dir
        rotation = jnp.stack([forward, jnp.array([-forward[1], forward[0]])])
        goal = s.goal_pos - s.agent_pos
        distance = jnp.linalg.norm(goal)
        distances = jnp.linalg.norm(s.path_array - s.agent_pos, axis=-1)
        nearest = jnp.argmin(distances)
        target = s.path_array[jnp.minimum(nearest + 3, p.path_length - 1)] - s.agent_pos
        direction = target / jnp.maximum(jnp.linalg.norm(target), 1e-6)
        obstacles = jnp.concatenate([s.static_obstacles, s.kinematic_obstacles], axis=0)
        rays = BaseEnv._generate_rays(forward, p.num_ray_sensors, p.agent_FOV)
        if obstacles.shape[0]:
            lengths = BaseEnv._ray_circle_intersections(s.agent_pos, rays, obstacles, 1.)
        else:
            lengths = jnp.full((p.num_ray_sensors,), jnp.inf)
        # Sensors also see the boundary; origins and distances are world units.
        upper = jnp.array([p.map_height_width[1] - 1., p.map_height_width[0] - 1.])
        safe_rays = jnp.where(jnp.abs(rays) > 1e-7, rays, 1e-7)
        wall_t = jnp.where(rays > 0., (upper - s.agent_pos) / safe_rays, -s.agent_pos / safe_rays)
        wall_t = jnp.where(jnp.abs(rays) > 1e-7, wall_t, jnp.inf)
        lengths = jnp.clip(jnp.minimum(lengths, jnp.min(wall_t, axis=-1)), 0., p.perception_radius)
        diagonal = float(np.hypot(*p.map_height_width))
        features = jnp.concatenate([rotation @ (goal / jnp.maximum(distance, 1e-6)),
                                    jnp.array([distance / diagonal, s.speed / p.max_speed, s.steering / p.max_steer]),
                                    rotation @ direction, jnp.array([distances[nearest] / diagonal]),
                                    lengths / p.perception_radius])
        return RoboTaxiObservation(features=features, distance_to_path=distances[nearest],
                                   direction_of_path=direction, collision_rays=rays * lengths[:, None])

    @staticmethod
    @partial(jax.jit, static_argnames=("params",))
    def _plan(start, goal, obstacles, params):
        """Bounded eight-neighbour Bellman relaxation with circle inflation.

        Moving objects are handled by perception, not frozen into the route.
        Navigation is a geometric hint, not a guarantee of bicycle feasibility.
        """
        h, w = params.map_height_width
        yy, xx = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
        cells = jnp.stack([xx, yy], axis=-1)
        gap = jnp.min(jnp.linalg.norm(cells[:, :, None, :] - obstacles, axis=-1), axis=-1, initial=jnp.inf)
        blocked = (gap <= 2.15) | (xx < 1) | (xx > w - 2) | (yy < 1) | (yy > h - 2)
        goal_cell = jnp.clip(jnp.rint(goal).astype(jnp.int32), jnp.array([1, 1]), jnp.array([w - 2, h - 2]))
        values = jnp.full((h, w), jnp.inf).at[goal_cell[1], goal_cell[0]].set(0.)
        moves = jnp.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1], [0, 0]])

        def neighbours(v, pos):
            nxt = pos[..., None, :] + moves
            nx, ny = nxt[..., 0], nxt[..., 1]
            valid = (nx >= 0) & (nx < w) & (ny >= 0) & (ny < h)
            # No diagonal corner cutting through blocked cells.
            valid &= ~blocked[pos[..., 1, None], jnp.clip(nx, 0, w - 1)]
            valid &= ~blocked[jnp.clip(ny, 0, h - 1), pos[..., 0, None]]
            score = v[jnp.clip(ny, 0, h - 1), jnp.clip(nx, 0, w - 1)] + jnp.linalg.norm(moves.astype(jnp.float32), axis=-1)
            return nxt, jnp.where(valid, score, jnp.inf)

        def relax(_, v):
            _, costs = neighbours(v, cells)
            return jnp.where(blocked, jnp.inf, jnp.minimum(v, jnp.min(costs, axis=-1)))

        values = jax.lax.fori_loop(0, h * w, relax, values)

        def trace(pos, _):
            nxt, costs = neighbours(values, pos)
            best = nxt[jnp.argmin(costs)]
            best = jnp.where(jnp.isfinite(jnp.min(costs)), best, pos)
            return best, best.astype(jnp.float32)

        start_cell = jnp.clip(jnp.rint(start).astype(jnp.int32), jnp.array([1, 1]), jnp.array([w - 2, h - 2]))
        _, path = jax.lax.scan(trace, start_cell, None, length=params.path_length)
        return path
