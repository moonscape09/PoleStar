#!/usr/bin/env python3
"""
collect_metaworld_data.py
Collects MetaWorld simulation images and saves them under ./data/metaworld_frames/<task>/

Usage:
    python collect_metaworld_data.py --task pick-place-v3 --episodes 20 --steps 200
"""

import os
import argparse

def collect_metaworld_frames(
    task_name="pick-place-v3",
    out_dir="./data/metaworld_frames",
    num_episodes=10,
    steps_per_ep=150,
    width=84,
    height=84,
    grayscale=False,
    frame_skip=1,
):
    """Collect and save frames from a MetaWorld environment."""
    try:
        import metaworld
        import gymnasium as gym
        import cv2
        import numpy as np
    except Exception as e:
        print("Missing dependency:", e)
        print("Run: pip install metaworld gymnasium mujoco opencv-python")
        return

    os.makedirs(out_dir, exist_ok=True)
    save_root = os.path.join(out_dir, task_name)
    os.makedirs(save_root, exist_ok=True)

    ml1 = metaworld.ML1(task_name)
    env = ml1.train_classes[task_name](render_mode="rgb_array")
    task = ml1.train_tasks[0]
    env.set_task(task)

    def render_img():
        img = env.render()
        img = cv2.resize(img, (width, height))
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    print(f"Collecting frames for {task_name}...")
    frame_count = 0
    for ep in range(num_episodes):
        _, _ = env.reset()
        ep_dir = os.path.join(save_root, f"ep_{ep:03d}")
        os.makedirs(ep_dir, exist_ok=True)
        for t in range(steps_per_ep):
            if t % frame_skip == 0:
                img = render_img()
                filename = os.path.join(ep_dir, f"frame_{t:04d}.png")
                if grayscale:
                    cv2.imwrite(filename, img)
                else:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(filename, img_bgr)
                frame_count += 1
            action = env.action_space.sample()
            _, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
    print(f"✅ Saved {frame_count} frames to {save_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect MetaWorld frames to ./data/")
    parser.add_argument("--task", type=str, default="pick-place-v3")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--width", type=int, default=84)
    parser.add_argument("--height", type=int, default=84)
    parser.add_argument("--gray", type=int, default=1)
    parser.add_argument("--frame_skip", type=int, default=1)
    args = parser.parse_args()

    collect_metaworld_frames(
        task_name=args.task,
        out_dir="./data/metaworld_frames",
        num_episodes=args.episodes,
        steps_per_ep=args.steps,
        width=args.width,
        height=args.height,
        grayscale=bool(args.gray),
        frame_skip=args.frame_skip,
    )
