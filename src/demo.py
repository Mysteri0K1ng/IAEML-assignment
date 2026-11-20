"""Headless MP4/GIF recording or interactive pygame control of the taxi."""
import argparse
from functools import partial
import json
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import jax
import jax.numpy as jnp
import numpy as np

from algorithms.ppo import network
from env.robotaxi import RoboTaxiEnv
from evaluate import heuristic, make_environment


def frame(state, obs, params, title, reward):
    width, height = 960, 800
    scale = min((width - 100) / (params.map_height_width[1] - 1), (height - 180) / (params.map_height_width[0] - 1))
    image = Image.new("RGB", (width, height), "#101827")
    draw = ImageDraw.Draw(image)
    # Pillow's bundled default font avoids platform-specific font paths.
    font, small = ImageFont.load_default(size=24), ImageFont.load_default(size=17)
    draw.text((36, 22), title, fill="#f1f5f9", font=font)
    draw.text((36, 59), f"t={float(state.time) / params.fps:.1f}s    speed={float(state.speed):.2f} m/s    return={reward:.1f}", fill="#a6bad3", font=small)

    def point(pos):
        return (50 + float(pos[0]) * scale, height - 60 - float(pos[1]) * scale)

    def circle(pos, radius, fill, outline=None):
        x, y = point(pos)
        r = radius * scale
        draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=outline, width=2)

    lower, upper = point([0, 0]), point([params.map_height_width[1] - 1, params.map_height_width[0] - 1])
    draw.rectangle([lower[0], upper[1], upper[0], lower[1]], outline="#61718a", width=2)
    path = [point(p) for p in np.asarray(state.path_array)]
    if len(path) > 1:
        draw.line(path, fill="#334b69", width=3)
    for obstacle in np.asarray(state.static_obstacles):
        circle(obstacle, 1., "#718096", "#bbc6d5")
    for obstacle, velocity in zip(np.asarray(state.kinematic_obstacles), np.asarray(state.kinematic_obst_velocities)):
        circle(obstacle, 1., "#ef7762", "#ffc4b8")
        draw.line([point(obstacle), point(obstacle + velocity)], fill="#ffc4b8", width=3)
    circle(state.goal_pos, 1., "#208b70", "#78edc3")
    draw.text((point(state.goal_pos)[0] - 20, point(state.goal_pos)[1] - 8), "GOAL", fill="white", font=small)
    for ray in np.asarray(obs.collision_rays):
        draw.line([point(state.agent_pos), point(state.agent_pos + ray)], fill="#28434d", width=1)
    circle(state.agent_pos, 1., "#58a6ff", "#bce0ff")
    draw.line([point(state.agent_pos), point(state.agent_pos + state.agent_forward_dir * 1.2)], fill="#ffffff", width=4)
    if bool(state.terminated) or bool(state.truncated):
        status = "GOAL REACHED" if bool(state.success) else "COLLISION" if bool(state.collision) else "TIME LIMIT"
        draw.text((36, 94), status, fill="#78edc3" if bool(state.success) else "#ff9c8a", font=font)
    draw.text((36, height - 30), "Blue: taxi   Green: destination   Gray: static   Coral: moving   Radius: 1 m", fill="#a6bad3", font=small)
    return np.asarray(image)


def run(checkpoint=None, policy="heuristic", seed=0, map_id=None, output="results/environment.mp4", interactive=False, fixed_start=False):
    if policy == "learned" and checkpoint is None:
        raise ValueError("Learned policy needs --checkpoint")
    weights, p, template = make_environment(checkpoint, map_id, randomized=not fixed_start)
    key = jax.random.PRNGKey(seed)
    obs, state = RoboTaxiEnv.reset(key, p, template)
    get_action = jax.jit(lambda obs: jnp.tanh(network(weights, obs.as_vector())[0])) if weights else None
    controller = jax.jit(lambda state: heuristic(state, p))
    writer = None
    total = 0.
    if interactive:
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((960, 800))
        pygame.display.set_caption("Robo-taxi | W/S throttle/brake | A/D steer | R reset | Esc exit")
        clock = pygame.time.Clock()
    else:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(path), fps=p.fps, codec="libx264", quality=8, macro_block_size=16) if path.suffix == ".mp4" else imageio.get_writer(str(path), mode="I", duration=1000 / p.fps, loop=0)
    title = f"IAEML Robo-taxi | {policy.upper()} | seed {seed}"
    try:
        for _ in range(p.max_steps_in_episode + 1):
            pixels = frame(state, obs, p, title, total)
            if writer:
                writer.append_data(pixels)
            if interactive:
                screen.blit(pygame.surfarray.make_surface(pixels.transpose(1, 0, 2)), (0, 0))
                pygame.display.flip()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                keys = pygame.key.get_pressed()
                if keys[pygame.K_ESCAPE]:
                    return
                if keys[pygame.K_r]:
                    key, kr = jax.random.split(key)
                    obs, state = RoboTaxiEnv.reset(kr, p, template)
                    total = 0.
                action = jnp.array([float(keys[pygame.K_w] or keys[pygame.K_UP]) - float(keys[pygame.K_s] or keys[pygame.K_DOWN]),
                                    float(keys[pygame.K_a] or keys[pygame.K_LEFT]) - float(keys[pygame.K_d] or keys[pygame.K_RIGHT])])
                clock.tick(p.fps)
                if bool(state.terminated | state.truncated):
                    continue
            else:
                if bool(state.terminated | state.truncated):
                    for _ in range(p.fps):
                        writer.append_data(pixels)
                    break
                key, ka = jax.random.split(key)
                action = get_action(obs) if policy == "learned" else controller(state) if policy == "heuristic" else jax.random.uniform(ka, (2,), minval=-1., maxval=1.)
            obs, state, reward, done, info = RoboTaxiEnv.step(key, state, action, p)
            total += float(reward)
    finally:
        if writer:
            writer.close()
        if interactive:
            pygame.quit()
    result = {"policy": policy, "seed": seed, "steps": int(state.time), "return": total,
              "success": bool(state.success), "collision": bool(state.collision), "truncated": bool(state.truncated)}
    if not interactive:
        Path(output).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
        Image.fromarray(frame(state, obs, p, title, total)).save(Path(output).with_suffix(".png"))
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--policy", choices=["learned", "heuristic", "random"], default="heuristic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--map-id", type=int)
    parser.add_argument("--output", default="results/environment.mp4")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--fixed-start", action="store_true")
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
