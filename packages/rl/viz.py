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
    parser.add_argument('-n', '--num_envs', type=int, default=None)
    parser.add_argument('--timeout', type=int, default=None)
    args = parser.parse_args()

    wandb.init(mode='disabled')
    gin_util.add_pytorch_external_configurables()
    gin_util.load_config_dict(os.path.join(args.logdir, 'config.json'))
    if args.num_envs is not None:
        gin_util.apply_bindings_from_dict({'envs.IsaacMixedXMLEnv.num_envs': args.num_envs})
    alg = gin.query_parameter('rl.train.algorithm').configurable.wrapped(args.logdir)
    alg.load(args.ckpt)
    env = TimeoutWrapper(alg.env, args.timeout) if args.timeout else alg.env
    env.finalize_obs_norm()
    env.create_viewer()
    rl_evaluate(env, alg.pi, nepisodes=alg.env.num_envs, t=alg.t, record_viewer=False)

