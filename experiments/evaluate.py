import argparse
import yaml
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.dqn_agent import DQNAgent
from agent.exploration_strategies import create_exploration_strategy
from environment.env_wrapper import create_environment
from utils.logger import ExperimentRunner

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_training_logs(log_dir: str, strategy: str, seed: int) -> pd.DataFrame:
    """Load training logs from CSV."""
    log_file = os.path.join(log_dir, f"{strategy}_seed{seed}.csv")
    if os.path.exists(log_file):
        return pd.read_csv(log_file)
    else:
        return None

def evaluate_strategy(env, agent, config, num_episodes=100):
    """Evaluate agent performance."""
    agent.set_training_mode(False)
    
    total_reward = 0.0
    episode_rewards = []
    successes = 0
    convergence_episode = -1
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        
        for step in range(config['training']['max_steps']):
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
            
            if done:
                if convergence_episode == -1:
                    convergence_episode = episode
                successes += 1
                break
        
        episode_rewards.append(episode_reward)
        total_reward += episode_reward
    
    return {
        "avg_reward": total_reward / num_episodes,
        "success_rate": (successes / num_episodes) * 100,
        "episode_rewards": episode_rewards,
        "convergence_episode": convergence_episode if convergence_episode != -1 else num_episodes
    }

def plot_learning_curves(log_dir: str, plot_dir: str, strategies: list):
    """Plot learning curves for all strategies."""
    plt.figure(figsize=(12, 7))
    
    for strategy in strategies:
        # Collect all seed results
        all_rewards = []
        all_episodes = None
        
        for seed in range(5):
            df = load_training_logs(log_dir, strategy, seed)
            if df is not None:
                all_rewards.append(df['reward'].values)
                if all_episodes is None:
                    all_episodes = df['episode'].values
        
        if all_rewards and all_episodes is not None:
            # Compute mean and std
            all_rewards = np.array(all_rewards)
            mean_rewards = np.mean(all_rewards, axis=0)
            std_rewards = np.std(all_rewards, axis=0)
            
            # Plot with confidence band
            plt.plot(all_episodes, mean_rewards, linewidth=2.5, label=strategy)
            plt.fill_between(all_episodes, 
                           mean_rewards - std_rewards,
                           mean_rewards + std_rewards,
                           alpha=0.2)
    
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.title("Learning Curves: Exploration Strategies Comparison", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "learning_curves.png"), dpi=300)
    print(f"✓ Saved plot: {os.path.join(plot_dir, 'learning_curves.png')}")
    plt.close()

def plot_success_rates(results: dict, plot_dir: str):
    """Plot success rates comparison."""
    strategies = list(results.keys())
    success_rates = [results[s]['success_rate'] for s in strategies]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(strategies, success_rates, color='steelblue', edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.ylabel("Success Rate (%)", fontsize=12)
    plt.title("Evaluation: Success Rate Comparison", fontsize=14)
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "success_rates.png"), dpi=300)
    print(f"✓ Saved plot: {os.path.join(plot_dir, 'success_rates.png')}")
    plt.close()

def plot_reward_comparison(results: dict, plot_dir: str):
    """Plot average reward comparison."""
    strategies = list(results.keys())
    avg_rewards = [results[s]['avg_reward'] for s in strategies]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(strategies, avg_rewards, color='coral', edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.ylabel("Average Reward", fontsize=12)
    plt.title("Evaluation: Average Reward Comparison", fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "avg_rewards.png"), dpi=300)
    print(f"✓ Saved plot: {os.path.join(plot_dir, 'avg_rewards.png')}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate DQN agents")
    parser.add_argument(
        "--env",
        type=str,
        default="FrozenLake-v1",
        help="Environment name"
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=5,
        help="Number of seeds used in training"
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        help="Generate comparison plots"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    config['environment']['name'] = args.env
    
    strategies = ["epsilon-greedy", "decaying-epsilon", "boltzmann", "ucb", "noisy-networks"]
    
    # Create environment
    env = create_environment(config['environment']['name'])
    
    results = {}
    
    print(f"\nEvaluating strategies on {config['environment']['name']}...\n")
    
    for strategy in strategies:
        print(f"Evaluating {strategy}...")
        
        strategy_results = []
        
        for seed in range(args.num_seeds):
            # Create agent
            strategy_config = config['exploration_strategies'].get(strategy, {})
            exploration_strategy = create_exploration_strategy(
                strategy, strategy_config, env.action_size
            )
            
            agent = DQNAgent(
                state_size=env.state_size,
                action_size=env.action_size,
                exploration_strategy=exploration_strategy,
                device=config['compute']['device']
            )
            
            # Load model
            model_path = os.path.join(
                config['logging']['model_dir'],
                f"model_{strategy}_seed{seed}.pt"
            )
            
            if os.path.exists(model_path):
                agent.load_model(model_path)
                result = evaluate_strategy(env, agent, config, args.eval_episodes)
                strategy_results.append(result)
            else:
                print(f"  ⚠ Model not found: {model_path}")
        
        if strategy_results:
            # Average across seeds
            avg_reward = np.mean([r['avg_reward'] for r in strategy_results])
            avg_success = np.mean([r['success_rate'] for r in strategy_results])
            
            results[strategy] = {
                'avg_reward': avg_reward,
                'success_rate': avg_success
            }
            
            print(f"  ✓ Avg Reward: {avg_reward:.4f}, Success: {avg_success:.2f}%\n")
    
    env.close()
    
    # Print summary table
    print("="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"{'Strategy':<20} {'Avg Reward':<20} {'Success Rate (%)':<20}")
    print("-"*70)
    
    for strategy in strategies:
        if strategy in results:
            r = results[strategy]
            print(f"{strategy:<20} {r['avg_reward']:<20.4f} {r['success_rate']:<20.2f}")
    
    print("="*70)
    
    # Generate plots
    if args.generate_plots:
        print("\nGenerating comparison plots...")
        plot_dir = config['logging']['plot_dir']
        log_dir = config['logging']['log_dir']
        
        plot_learning_curves(log_dir, plot_dir, strategies)
        plot_success_rates(results, plot_dir)
        plot_reward_comparison(results, plot_dir)
        
        print(f"✓ Plots saved to {plot_dir}")


if __name__ == "__main__":
    main()