from utils import gin_util
from rl.trainer import train


@gin_util.wandb_main
def main():
    train()

if __name__ == '__main__':
    main()
