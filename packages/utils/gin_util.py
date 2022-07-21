"""Register pytorch classes and functions with gin."""
import gin
import inspect
import argparse
import json


def add_pytorch_external_configurables():
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import torchvision
    import mup

    optimizers = [obj for name, obj in inspect.getmembers(optim)
                  if inspect.isclass(obj)]
    for o in optimizers:
        gin.config.external_configurable(o, module='optim')

    modules = [obj for name, obj in inspect.getmembers(nn) if inspect.isclass(obj)]
    for m in modules:
        gin.config.external_configurable(m, module='nn')

    funcs = [f for name, f in inspect.getmembers(F) if inspect.isfunction(f)]
    for f in funcs:
        try:
            gin.config.external_configurable(f, module='F')
        except Exception:
            pass

    funcs = [f for name, f in inspect.getmembers(torchvision.models)
             if inspect.isfunction(f)]
    for f in funcs:
        try:
            gin.config.external_configurable(f, module='models')
        except Exception:
            pass

    funcs = [f for name, f in inspect.getmembers(torchvision.models.segmentation)
             if inspect.isfunction(f)]
    for f in funcs:
        try:
            gin.config.external_configurable(f, module='models.segmentation')
        except Exception:
            pass

    funcs = [f for name, f in inspect.getmembers(torchvision.models.detection)
             if inspect.isfunction(f)]
    for f in funcs:
        try:
            gin.config.external_configurable(f, module='models.detection')
        except Exception:
            pass

    transforms = [obj for name, obj in inspect.getmembers(torchvision.transforms)
                  if inspect.isclass(obj)]
    for t in transforms:
        try:
            gin.config.external_configurable(t, module='transforms')
        except Exception:
            pass

    datasets = [obj for name, obj in inspect.getmembers(torchvision.datasets)
                if inspect.isclass(obj)]
    for d in datasets:
        try:
            gin.config.external_configurable(d, module='datasets')
        except Exception:
            pass

    gin.config.external_configurable(mup.optim.MuSGD, module='mup')
    gin.config.external_configurable(mup.optim.MuAdam, module='mup')
    gin.config.external_configurable(mup.optim.MuAdamW, module='mup')


def load_config(gin_files, gin_bindings=[], finalize=True):
    """Load gin configuration files.

    Args:
    gin_files: path or list of paths to the gin configuration files for this
      experiment.
    gin_bindings: list, of gin parameter bindings to override the values in
      the config files.

    """
    if isinstance(gin_files, str):
        gin_files = [gin_files]
    gin.parse_config_files_and_bindings(gin_files,
                                        bindings=gin_bindings,
                                        skip_unknown=False,
                                        finalize_config=finalize)


def parse_gin_args():
    parser = argparse.ArgumentParser("gin-config parser")
    parser.add_argument('config', help='gin file')
    parser.add_argument('bindings', nargs='*', help='gin bindings', metavar='x=y a=b ...')
    return parser.parse_args()


def get_config_dict():
    config = {}
    for (scope, name), params in gin.config._CONFIG.items():
        cid = f'{scope}/{name}' if len(scope) > 0 else name
        for k, v in params.items():
            if isinstance(v, gin.config.ConfigurableReference):
                config[f'{cid}.{k}'] = str(v)
            else:
                config[f'{cid}.{k}'] = v
    return config


def apply_bindings_from_dict(config):
    bindings = []
    for k, v in config.items():
        if isinstance(v, str) and v[0] != '@':
            bindings.append(f'{k}="{v}"')
        else:
            bindings.append(f"{k}={v}")
    gin.config.parse_config(bindings)


def save_config_dict(filename):
    config = get_config_dict()
    with open(filename, 'w') as f:
        json.dump(config, f)


def load_config_dict(filename):
    with open(filename, 'r') as f:
        config = json.load(f)
    apply_bindings_from_dict(config)


def main(f):
    def main_fn(*args, **kwargs):
        gin_args = parse_gin_args()
        add_pytorch_external_configurables()
        load_config(gin_args.config, gin_args.bindings, finalize=True)
        f(*args, **kwargs)

    return main_fn



@gin.configurable(module='logging')
def wandb_init(**kwargs):
    """See wandb init documentation for args"""
    return kwargs


def wandb_main(f):
    import wandb

    def main_fn(*args, **kwargs):
        gin_args = parse_gin_args()
        add_pytorch_external_configurables()
        load_config(gin_args.config, gin_args.bindings, finalize=False)
        config = get_config_dict()

        with wandb.init(config=config, **wandb_init()):
            config = wandb.config
            apply_bindings_from_dict(config)
            gin.config.finalize()
            f(*args, **kwargs)

    return main_fn
