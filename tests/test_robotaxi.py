from dataclasses import replace
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from env.base import BaseEnv
from env.robotaxi import RoboTaxiEnv as Env, _replace


@pytest.fixture(scope="module")
def setup():
    return Env.init_params(jax.random.PRNGKey(0), randomize_reset=False)


def test_interface_and_transforms(setup):
    p, s = setup
    assert issubclass(Env, BaseEnv)
    obs, reset = Env.reset(jax.random.PRNGKey(0), p, s)
    assert obs.as_vector().shape == (24,)
    keys = jax.random.split(jax.random.PRNGKey(1), 4)
    batch_obs, states = jax.jit(jax.vmap(lambda k: Env.reset(k, p, s)))(keys)
    out = jax.jit(jax.vmap(lambda k, st: Env.step(k, st, jnp.array([1., .2]), p)))(keys, states)
    assert out[0].as_vector().shape == (4, 24)
    assert np.isfinite(out[0].as_vector()).all()


def test_acceleration_drag_and_steering(setup):
    p, s = setup
    _, driven, _, _, _ = Env.step(jax.random.PRNGKey(0), s, jnp.array([1., .5]), p)
    assert driven.speed > s.speed
    assert driven.agent_pos[0] > s.agent_pos[0]
    assert driven.agent_forward_dir[1] > 0
    _, coasted, _, _, _ = Env.step(jax.random.PRNGKey(0), driven, jnp.zeros(2), p)
    assert coasted.speed < driven.speed
    assert driven.steering <= p.steer_rate * p.step_size + 1e-6
    np.testing.assert_allclose(jnp.linalg.norm(driven.agent_forward_dir), 1., atol=1e-6)


def test_radius_sum_collision_and_absorbing(setup):
    p, s = setup
    s = _replace(s, static_obstacles=jnp.array([[5.9, 10.]]))
    _, end, reward, done, info = Env.step(jax.random.PRNGKey(0), s, jnp.zeros(2), p)
    assert done and info["collision"] and info["terminated"]
    assert reward < -14
    _, again, reward, done, _ = Env.step(jax.random.PRNGKey(1), end, jnp.ones(2), p)
    assert reward == 0 and done
    for a, b in zip(jax.tree.leaves(end), jax.tree.leaves(again)):
        np.testing.assert_array_equal(a, b)


def test_goal_and_time_limit_are_distinct(setup):
    p, s = setup
    goal = _replace(s, agent_pos=s.goal_pos)
    _, _, reward, done, info = Env.step(jax.random.PRNGKey(0), goal, jnp.zeros(2), p)
    assert done and info["success"] and not info["truncated"] and reward > 24
    _, _, _, done, info = Env.step(jax.random.PRNGKey(0), s, jnp.zeros(2), replace(p, max_steps_in_episode=1))
    assert done and info["truncated"] and not info["terminated"]


def test_swept_collision(setup):
    p, s = setup
    # Moving circle crosses the stationary car between integration endpoints.
    s = _replace(s, kinematic_obstacles=jnp.array([[1., 10.]]), kinematic_obst_velocities=jnp.array([[60., 0.]]))
    _, _, _, done, info = Env.step(jax.random.PRNGKey(0), s, jnp.zeros(2), replace(p, substeps=1))
    assert done and info["collision"]


def test_reset_reproducibility_and_variation(setup):
    p, s = setup
    p = replace(p, randomize_reset=True)
    a = Env.reset(jax.random.PRNGKey(5), p, s)[1]
    b = Env.reset(jax.random.PRNGKey(5), p, s)[1]
    c = Env.reset(jax.random.PRNGKey(6), p, s)[1]
    np.testing.assert_array_equal(a.agent_pos, b.agent_pos)
    assert not np.array_equal(a.agent_pos, c.agent_pos)
    assert Env._clearance(a.agent_pos, a.static_obstacles, a.kinematic_obstacles, p) > 0


def test_planner_does_not_wrap_and_clears_circles(setup):
    p, s = setup
    assert np.linalg.norm(np.asarray(s.path_array[-1] - s.goal_pos)) < 1e-5
    assert np.min(np.linalg.norm(np.asarray(s.path_array[:, None] - s.static_obstacles), axis=-1)) > 2.
    assert np.max(np.linalg.norm(np.diff(np.asarray(s.path_array), axis=0), axis=-1)) <= np.sqrt(2.) + 1e-5


def test_empty_obstacles_and_finite_clipped_actions(setup):
    p, s = setup
    s = _replace(s, static_obstacles=jnp.zeros((0, 2)), kinematic_obstacles=jnp.zeros((0, 2)), kinematic_obst_velocities=jnp.zeros((0, 2)))
    obs, _, reward, _, _ = Env.step(jax.random.PRNGKey(0), s, jnp.array([jnp.inf, jnp.nan]), p)
    assert np.isfinite(obs.as_vector()).all() and np.isfinite(reward)


def test_rays_measure_surface_distance():
    distances = BaseEnv._ray_circle_intersections(jnp.array([0., 0.]), jnp.array([[1., 0.], [-1., 0.]]), jnp.array([[4., 0.]]), 1.)
    np.testing.assert_allclose(distances[0], 3.)
    assert np.isinf(distances[1])


@pytest.mark.parametrize("map_id", [0, 1, 2])
def test_scaled_original_maps_have_valid_static_routes(map_id):
    p, s = Env.init_params(jax.random.PRNGKey(0), map_id=map_id, world_scale=3.)
    assert Env._clearance(s.agent_pos, s.static_obstacles, s.kinematic_obstacles, p) > 0
    np.testing.assert_allclose(s.path_array[-1], s.goal_pos, atol=1e-5)
    assert np.min(np.linalg.norm(np.asarray(s.path_array[:, None] - s.static_obstacles), axis=-1)) > 2.
