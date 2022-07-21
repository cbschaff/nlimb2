import argparse
import os
import gin
import wandb
from utils import gin_util
from rl.util.eval import rl_evaluate
from rl.wrappers.timeout_wrapper import TimeoutWrapper


def load_config_from_logdir(logdir):
    gin_util.add_pytorch_external_configurables()
    gin_util.load_config(os.path.join(logdir, 'config.gin'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir')
    parser.add_argument('--ckpt', '-t', type=int, default=None)
    parser.add_argument('--num_episodes', '-n', type=int)
    parser.add_argument('--timeout', default= None, type=int)
    parser.add_argument('--record', '-r', default=False, action='store_true')

    args = parser.parse_args()

    wandb.init(mode='disabled')

    load_config_from_logdir(args.logdir)
    alg = gin.query_parameter('rl.train.algorithm').configurable.wrapped(args.logdir)
    t = alg.load(args.ckpt)
    outdir = os.path.join(args.logdir, f'eval/{t:012d}')
    env = TimeoutWrapper(alg.env, args.timeout) if args.timeout else alg.env
    rl_evaluate(env, alg.pi, args.num_episodes, t, outdir, record_viewer=args.record)
