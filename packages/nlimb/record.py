import os
import argparse
import wandb
import gin
import envs
import nlimb
import tasks
from utils import gin_util
from rl.util import rl_evaluate
from rl.wrappers.timeout_wrapper import TimeoutWrapper


if __name__ == '__main__':
    parser = argparse.ArgumentParser('viz')
    parser.add_argument('logdir')
    parser.add_argument('-t', '--ckpt', type=int, default=None)
    args = parser.parse_args()

    wandb.init(mode='disabled')
    gin_util.add_pytorch_external_configurables()
    gin_util.load_config_dict(os.path.join(args.logdir, 'config.json'))
    config = gin_util.get_config_dict()
    config['envs.IsaacMixedXMLEnv.num_envs'] = 1
    config['envs.IsaacMixedXMLEnv.create_eval_sensors'] = True
    gin_util.apply_bindings_from_dict(config)
    alg = gin.query_parameter('rl.train.algorithm').configurable.wrapped(args.logdir)
    alg.load(args.ckpt)
    pi = alg.alg.pi
    env = TimeoutWrapper(alg.alg.env, alg.alg.eval_max_episode_length)
    env.finalize_obs_norm()
    env.init_scene(evaluate=True, mode=True)
    rl_evaluate(env, pi, nepisodes=alg.env.num_envs, t=alg.t, record_viewer=False,
                outdir=args.logdir, num_recording_envs=1)
