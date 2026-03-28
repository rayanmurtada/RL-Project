# Deep Q-Learning Exploration Strategies Comparison

A comprehensive research project implementing and comparing different exploration strategies in Deep Q-Learning (DQN) under sparse reward environments.

## 📋 Project Overview

This project investigates how different exploration strategies affect learning performance in reinforcement learning agents trained with DQN. We compare five exploration approaches:

1. **Epsilon-Greedy** (baseline)
2. **Decaying Epsilon-Greedy** 
3. **Boltzmann (Softmax) Exploration**
4. **Upper Confidence Bound (UCB) Exploration**
5. **Noisy Networks** (bonus implementation)

### 🎯 Research Hypothesis

**Advanced exploration strategies (Boltzmann, UCB, Noisy Networks) outperform standard epsilon-greedy exploration in sparse reward environments by improving sample efficiency, convergence stability, and success rates.**

### 📊 Research Questions

1. How do different exploration strategies affect sample efficiency in sparse reward settings?
2. Which strategy converges fastest to optimal or near-optimal policies?
3. How stable are the exploration strategies across multiple random seeds?
4. What is the trade-off between exploration and exploitation for each strategy?

---

## 🧪 Environments

The project supports two sparse reward environments from Gymnasium:

### FrozenLake-v1
- **State Space**: 16 discrete states (4×4 grid)
- **Action Space**: 4 actions (up, down, left, right)
- **Reward Structure**: +1 for reaching goal, 0 otherwise (sparse)
- **Difficulty**: Non-slippery floor for deterministic transitions

### MountainCar-v0
- **State Space**: 2 continuous variables (position, velocity)
- **Action Space**: 3 actions (left, neutral, right)
- **Reward Structure**: -1 per step, goal bonus (sparse)
- **Challenge**: Requires exploration to escape valley

---

## 🏗️ Architecture

### Project Structure

```
dqn-exploration-strategies/
├── agent/
│   ├── dqn_agent.py              # DQN agent implementation with target network
│   └── exploration_strategies.py # 5 exploration strategy implementations
├── environment/
│   └── env_wrapper.py            # Gymnasium environment wrapper
├── utils/
│   ├── replay_buffer.py          # Experience replay buffer
│   └── logger.py                 # Training metrics logging
├── experiments/
│   ├── train.py                  # Training script with CLI
│   └── evaluate.py               # Evaluation and plotting
├── results/
│   ├── plots/                    # Generated visualizations
│   ├── logs/                     # Training logs (CSV)
│   └── models/                   # Saved trained models
├── config.yaml                   # Hyperparameter configuration
├── requirements.txt              # Project dependencies
└── README.md                     # This file
```

### Neural Network Architecture

```
Input (State)
    ↓
Dense Layer (128 units, ReLU)
    ↓
Dense Layer (128 units, ReLU)
    ↓
Output Layer (Action Q-values)
```

- **Input Dimension**: 16 (FrozenLake one-hot encoded) or 2 (MountainCar continuous)
- **Hidden Layers**: 128 units each with ReLU activation
- **Output Dimension**: 4 (FrozenLake) or 3 (MountainCar)

### Key Training Components

1. **Experience Replay Buffer**: Stores and samples transitions for decorrelated training
2. **Target Network**: Separate network updated every N episodes for stable targets
3. **Loss Function**: Mean Squared Error (MSE) between predicted and target Q-values
4. **Optimizer**: Adam with learning rate 0.001

---

## 🚀 Installation & Setup

### Requirements

- Python 3.8+
- PyTorch 2.0+
- Gymnasium 0.28+
- NumPy, Matplotlib, Pandas, PyYAML

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/rayanmurtada/RL-Project.git
cd RL-Project

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')
import gymnasium; print(f'Gymnasium version: {gymnasium.__version__}')"
```

---

## 📖 How to Run

### 1. Train Single Strategy

Train a specific exploration strategy:

```bash
# Train epsilon-greedy on FrozenLake
python experiments/train.py --strategy epsilon-greedy --env FrozenLake --episodes 1000 --seed 42

# Train Boltzmann on MountainCar
python experiments/train.py --strategy boltzmann --env MountainCar --episodes 2000 --seed 123

# Train UCB exploration
python experiments/train.py --strategy ucb --env FrozenLake --episodes 1000
```

### 2. Train All Strategies (Multi-Seed)

Train all exploration strategies with multiple random seeds for statistical reliability:

```bash
# Train all strategies with 5 seeds each
python experiments/train.py --train-all --env FrozenLake --num-seeds 5 --episodes 1000

# Train all strategies on MountainCar
python experiments/train.py --train-all --env MountainCar --num-seeds 3 --episodes 2000
```

### 3. Evaluate and Compare

Generate comparison plots and summary statistics:

```bash
# Evaluate all strategies
python experiments/evaluate.py --env FrozenLake --num-seeds 5 --generate-plots

# Evaluate specific strategies
python experiments/evaluate.py --env FrozenLake --strategies epsilon-greedy decaying-epsilon boltzmann --generate-plots
```

### 4. Command-Line Interface Reference

```
Training Options:
  --strategy {epsilon-greedy, decaying-epsilon, boltzmann, ucb, noisy-networks}
  --env {FrozenLake, MountainCar}
  --episodes NUM          Number of training episodes (default: 1000)
  --seed NUM              Random seed (default: 42)
  --train-all             Train all strategies
  --num-seeds NUM         Number of seeds for multi-seed training (default: 3)
  --config FILE           Path to config file (default: config.yaml)
  --no-save               Do not save trained models

Evaluation Options:
  --env ENV               Environment name (default: FrozenLake)
  --num-seeds NUM         Number of seeds to aggregate (default: 3)
  --generate-plots        Generate comparison plots
  --strategies NAMES      Specific strategies to evaluate
```

---

## 🔧 Configuration

Edit `config.yaml` to modify hyperparameters:

```yaml
# Environment Settings
environment:
  name: "FrozenLake"           # Or "MountainCar"
  is_slippery: false           # For FrozenLake
  max_steps: 100               # Max steps per episode

# Network Architecture
network:
  hidden_dim_1: 128            # First hidden layer size
  hidden_dim_2: 128            # Second hidden layer size

# Training Parameters
training:
  num_episodes: 1000           # Total episodes
  batch_size: 32               # Mini-batch size
  learning_rate: 0.001         # Adam learning rate
  gamma: 0.99                  # Discount factor
  target_update_freq: 10       # Update target network every N episodes

# Exploration Strategy Parameters
exploration:
  epsilon_greedy:
    epsilon: 0.1               # Fixed exploration rate  
  
  decaying_epsilon:
    epsilon_start: 1.0         # Initial epsilon
    epsilon_end: 0.01          # Minimum epsilon
    decay_rate: 0.995          # Decay per episode  
  
  boltzmann:
    temperature_start: 1.0     # Initial temperature
    temperature_end: 0.01      # Minimum temperature
    temperature_decay: 0.995  
  
  ucb:
    exploration_weight: 1.0    # UCB exploration bonus weight  
  
  noisy_networks:
    sigma_init: 0.5            # Initial noise std dev
```

---

## 📊 Evaluation Metrics

### Primary Metrics

1. **Average Reward**: Mean cumulative reward per episode
2. **Success Rate**: Percentage of episodes reaching goal (reward > 0)
3. **Convergence Speed**: Episodes to reach 50% success rate
4. **Stability**: Standard deviation across multiple seeds

### Output Files

Training generates:
- **CSV Logs**: `results/logs/{strategy}_seed_{n}_metrics.csv`
  - Columns: episode, reward, loss, success, exploration_value
- **Trained Models**: `results/models/{strategy}_seed_{n}.pt`
- **Plots**: `results/plots/` (PNG format, 300 DPI)

---

## 📈 Results & Findings

### Expected Results (Preliminary Hypothesis)

Based on reinforcement learning theory, we expect:

| Strategy | Convergence | Stability | Sample Efficiency |
|----------|-----------|-----------|------------------|
| Epsilon-Greedy | Moderate | Low | Low |
| Decaying-Epsilon | Good | Moderate | Moderate |
| Boltzmann | Very Good | High | High |
| UCB | Excellent | Very High | Very High |
| Noisy Networks | Excellent | Very High | Very High |

### Plotting Output

The evaluation script generates:

1. **strategy_comparison_rewards.png** - All strategies' reward curves with confidence bands
2. **strategy_comparison_success.png** - Success rate comparison
3. **individual_{strategy}.png** - Individual strategy analysis across seeds

Each plot shows:
- Mean performance (bold line)
- Standard error bands (shaded region)
- Individual seed runs (thin lines)

### Example Commands for Full Analysis

```bash
# Complete training pipeline
python experiments/train.py --train-all --env FrozenLake --num-seeds 5 --episodes 1000

# Generate comprehensive comparison
python experiments/evaluate.py --env FrozenLake --num-seeds 5 --generate-plots
```

---

## 🔬 Technical Details

### Exploration Strategies Implemented

#### 1. Epsilon-Greedy
```
With probability ε: select random action
Otherwise: select argmax(Q(s, a))
```
- **Fixed epsilon**: No decay
- **Use Case**: Baseline comparison

#### 2. Decaying Epsilon-Greedy
```
ε_t = ε_end + (ε_start - ε_end) × decay_rate^t
```
- Gradually shifts from exploration to exploitation
- **Benefit**: Automatic balance between exploration and exploitation

#### 3. Boltzmann (Softmax) Exploration
```
π(a|s) = exp(Q(s,a) / τ) / Σ_a' exp(Q(s,a') / τ)
τ_t = τ_end + (τ_start - τ_end) × decay_rate^t
```
- Probabilistic action selection based on Q-values
- **Benefit**: Smooth transition between actions, considers all Q-values

#### 4. UCB Exploration
```
UCB(a) = Q(a) + c × √(ln(t) / N(a))
```
- Balances exploitation (Q-value) with exploration bonus
- **Benefit**: Theoretically-grounded optimism under uncertainty

#### 5. Noisy Networks
```
w = μ_w + σ_w ⊙ ε_w  (element-wise product)
```
- Adds learnable noise to network weights
- **Benefit**: Systematic, state-dependent exploration; no hyperparameter tuning

---

## 🎓 Learning Outcomes

This project demonstrates:

1. **DQN Fundamentals**: Implementation of core DQN components
2. **Exploration-Exploitation Trade-off**: Practical comparison of strategies
3. **Experimental Design**: Fair comparison with proper controls (same seeds, hyperparameters)
4. **Statistical Analysis**: Multi-seed aggregation and confidence intervals
5. **Reproducibility**: Seed management and configuration-driven training

---

## 📚 References

### Key Papers

1. Mnih et al. (2015) - "Human-level control through deep reinforcement learning"
   - Introduces DQN with experience replay and target networks

2. Chua et al. (2018) - "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models"
   - Discussion of exploration strategies in model-based RL

3. Fortunato et al. (2017) - "Noisy Networks for Exploration"
   - Noisy Networks exploration approach

### Gymnasium Documentation
- https://gymnasium.readthedocs.io/

### PyTorch Documentation
- https://pytorch.org/docs/

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'gymnasium'"
**Solution**: `pip install gymnasium`

### Issue: "CUDA out of memory" or GPU errors
**Solution**: The project defaults to CPU. For GPU training, modify agent creation:
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### Issue: "No plots generated"
**Solution**: Ensure `--generate-plots` flag is used and results exist:
```bash
python experiments/evaluate.py --env FrozenLake --num-seeds 3 --generate-plots
```

### Issue: Training is very slow
**Solution**: 
- Reduce `--episodes` for testing
- Use fewer `--num-seeds` initially
- Check that batch size matches replay buffer capacity

---

## 📝 Future Enhancements

Potential extensions:

1. **Double DQN**: Reduce overestimation bias
2. **Dueling DQN**: Separate value and advantage streams
3. **Prioritized Experience Replay**: Weight important transitions
4. **Multi-Step Returns**: Bootstrap further into future
5. **Additional Environments**: Atari games, robotic control
6. **Visualization**: T-SNE plots of learned state representations
7. **Hyperparameter Tuning**: Automated search via Optuna/Ray Tune

---

## 📄 License

This project is provided for educational and research purposes.

---

## 👨‍💻 Author

**Rayan Murtada** - Reinforcement Learning Research

For questions or issues, please open a GitHub issue.

---

## 🙏 Acknowledgments

- OpenAI for Gym/Gymnasium
- Meta for PyTorch
- The reinforcement learning community for foundational algorithms

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{murtada2024dqn_exploration,
  author={Murtada, Rayan},
  title={Deep Q-Learning Exploration Strategies Comparison},
  year={2024},
  url={https://github.com/rayanmurtada/RL-Project}
}
```

---

**Last Updated**: 2026-03-28 10:28:44
**Status**: Active Development
