import argparse

def main(args):
    print(f'Strategy: {args.strategy}')
    print(f'Environment: {args.env}')
    print(f'Episodes: {args.episodes}')
    print(f'Train All: {args.train_all}')
    print(f'Number of Seeds: {args.num_seeds}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script for experiments.')
    parser.add_argument('--strategy', type=str, required=True, help='The strategy to use for training')
    parser.add_argument('--env', type=str, required=True, help='The environment to train in')
    parser.add_argument('--episodes', type=int, required=True, help='Number of episodes to train')
    parser.add_argument('--train-all', action='store_true', help='Whether to train all strategies')
    parser.add_argument('--num-seeds', type=int, required=True, help='Number of seeds for training')
    args = parser.parse_args()
    main(args)
