"""Evaluation for RL Environments."""
import os
import json
from subprocess import call

import torch
import numpy as np
import wandb
from utils import misc


def rl_evaluate(env, pi, nepisodes, t, outdir=None, num_recording_envs=0, record_viewer=True,
                log_to_wandb=False):
    """Record episode stats."""

    if nepisodes == 0:
        return

    if log_to_wandb:
        wandb.define_metric("eval/*", step_metric="train/step")

    misc.set_env_to_eval_mode(env)
    pi.eval()
    episode_lengths = []
    episode_rewards = []
    obs = env.reset()

    # record
    if record_viewer:
        env.record_viewer(os.path.join(outdir, 'viewer.mp4'))
    num_recording_envs = min(num_recording_envs, env.num_envs)
    for i in range(num_recording_envs):
        env.record_env(i, os.path.join(outdir, f'env{i}.mp4'))

    while len(episode_lengths) < nepisodes:
        obs = env.reset()
        batch_dones = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        batch_length = torch.zeros(env.num_envs, device=env.device, dtype=torch.float)
        batch_reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float)
        while torch.any(batch_dones):
            with torch.no_grad():
                ac = pi(obs).action
                obs, rews, dones, _ = env.step(ac)
            assert dones.shape == batch_dones.shape
            batch_length[batch_dones] += 1
            batch_reward[batch_dones] += rews[batch_dones]
            torch.logical_and(torch.logical_not(dones), batch_dones, out=batch_dones)
        episode_lengths += batch_length.cpu().numpy().tolist()
        episode_rewards += batch_reward.cpu().numpy().tolist()

    data = {
        'episode_lengths': episode_lengths,
        'episode_rewards': episode_rewards,
        'mean_length': float(np.mean(episode_lengths)),
        'mean_reward': float(np.mean(episode_rewards)),
        'median_length': float(np.median(episode_lengths)),
        'median_reward': float(np.median(episode_rewards)),
    }

    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, 'data.txt'), 'w') as f:
            json.dump(data, f)

    if log_to_wandb:
        wandb.log({'eval/mean_episode_rewards': data['mean_reward'],
                   'eval/mean_episode_lengths': data['mean_length'],
                   'eval/median_episode_rewards': data['median_reward'],
                   'eval/median_episode_lengths': data['median_length'],
                   'train/step': t})

    # write videos
    if record_viewer:
        outfile = os.path.join(outdir, 'viewer.mp4')
        env.write_viewer_video()
        if log_to_wandb:
            wandb.log({'eval/viewer': wandb.Video(outfile), 'train/step': t})

    fnames = []
    for i in range(num_recording_envs):
        outfile = os.path.join(outdir, f'env{i}.mp4')
        env.write_env_video(i)
        fnames.append(outfile)

    if num_recording_envs > 0:
        outfile = os.path.join(outdir, 'all_envs.mp4')
        merge_cmd = ['ffmpeg']
        for fname in fnames:
            merge_cmd += ['-i', fname]
        merge_cmd += ['-filter_complex', 'hstack', outfile]
        call(merge_cmd)
        if log_to_wandb:
            wandb.log({'eval/envs': wandb.Video(outfile), 'train/step': t})

    env.reset()
    misc.set_env_to_train_mode(env)
    pi.train()
    return data
