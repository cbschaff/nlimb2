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
    args = parser.parse_args()

    wandb.init(mode='disabled')
    gin_util.add_pytorch_external_configurables()
    gin_util.load_config_dict(os.path.join(args.logdir, 'config.json'))
    if args.num_envs is not None:
        config = gin_util.get_config_dict()
        config['envs.IsaacMixedXMLEnv.num_envs'] = args.num_envs
        config['envs.IsaacMixedXMLEnv.spacing'] = (0., 0.025, 1.)
        gin_util.apply_bindings_from_dict(config)
    alg = gin.query_parameter('rl.train.algorithm').configurable.wrapped(args.logdir)
    alg.load(args.ckpt)
    pi = alg.alg.pi
    env = alg.alg.env
    env.finalize_obs_norm()
    # env = TimeoutWrapper(env, 2*alg.alg.eval_max_episode_length)
    env.init_scene(evaluate=True, mode=True)
    env.create_viewer()
    rl_evaluate(env, pi, nepisodes=alg.env.num_envs, t=alg.t, record_viewer=False)

