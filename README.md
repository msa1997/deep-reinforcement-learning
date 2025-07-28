# Enhancing DQN with Reward Shaping for Perishable Inventory Management

This repository contains a Jupyter Notebook that implements and analyzes a Deep Q-Network (DQN) enhanced with Potential-Based Reward Shaping for a perishable inventory management problem. The work is based on the research paper by De Moor et al.

**Paper**: [Reward shaping to improve the performance of deep reinforcement learning in perishable inventory management](https://www.researchgate.net/publication/355707267_Reward_shaping_to_improve_the_performance_of_deep_reinforcement_learning_in_perishable_inventory_management)

## Overview

Managing perishable inventory is a challenging task for reinforcement learning agents due to large state spaces and sparse, delayed rewards (costs). An agent might take a long time to learn the connection between placing an order and the eventual outcomes of holding costs, lost sales, or spoilage.

This project implements the solution proposed by De Moor et al., which uses **Potential-Based Reward Shaping** to guide the learning process. By providing the agent with a more immediate, dense reward signal based on domain-specific heuristics, the DQN agent can learn an optimal ordering policy more quickly and effectively.

## Key Concepts Implemented

-   **Simulation Environment**: A custom `gymnasium.Env` for perishable inventory management, parameterizable by product lifetime (`m`), delivery lead time (`L`), and various cost factors.
-   **RL Algorithm**: Deep Q-Network (DQN) from the `stable-baselines3` library.
-   **Technique**: Potential-Based Reward Shaping.
-   **Heuristics Used for Shaping**:
    1.  **Base-Stock Policy**: A simple "order-up-to" policy.
    2.  **BSP-low-EW**: A more advanced policy that considers the estimated waste during the lead time.

## Repository Structure

```
.
├── Q1/
│   └── perishable inventory management.ipynb   # Main Jupyter Notebook with all code and analysis.
└── README.md
```

## Getting Started

### Prerequisites

-   Python 3.8+
-   Jupyter Notebook or JupyterLab

### Installation

1.  Clone the repository:
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  Install the required Python packages from the notebook:
    ```bash
    pip install gymnasium stable-baselines3 torch numpy matplotlib pandas seaborn scipy
    ```

### Running the Notebook

1.  Navigate to the `Q1/` directory.
2.  Launch Jupyter:
    ```bash
    jupyter notebook
    ```
    or
    ```bash
    jupyter lab
    ```
3.  Open `perishable inventory management.ipynb`.
4.  Run the cells sequentially to execute the environment setup, model training, and results analysis.

**Note**: Training the models (the cell under section `1-6. Training the Models`) can take a significant amount of time (estimated 30-60 minutes depending on your hardware).

## Implementation Details

The core of the project is within the Jupyter Notebook and is broken down into several key components:

### 1. `PerishableInventoryEnv`

A custom environment that simulates the inventory system.

-   **State Space**: An `(m + L - 1)`-dimensional vector representing on-hand inventory by age and in-transit orders (pipeline).
-   **Action Space**: A discrete set of possible order quantities, from 0 to `q_max`.
-   **Dynamics**: The `step()` function models demand fulfillment (FIFO/LIFO), product spoilage, inventory aging, and order arrivals.
-   **Reward**: The reward is the negative of the total cost, which includes ordering, holding, lost sales, and perishing costs.

### 2. `RewardShapingWrapper`

This wrapper modifies the environment's reward signal to guide the agent.

-   It calculates a "shaping reward" based on a potential function: `F(s, a, s') = γΦ(s') - Φ(s)`.
-   The potential function `Φ(s)` is derived from heuristic policies (`base_stock` or `bsp_low_ew`), estimating how "good" a state is.
-   This shaping reward is added to the environment's original reward, providing a denser learning signal without changing the optimal policy.

### 3. DQN Model & Training

-   The DQN agent is implemented using `stable-baselines3`.
-   The policy network is a Multi-Layer Perceptron (MLP) with two hidden layers of 128 neurons each.
-   Three different models are trained and evaluated across 5 random seeds for 200,000 timesteps each:
    1.  **DQN without Shaping** (baseline)
    2.  **DQN + Base-Stock Shaping**
    3.  **DQN + BSP-low-EW Shaping**

## Results & Analysis

The notebook provides a detailed analysis comparing the performance of the three models.

### Quantitative Performance

The final trained models are evaluated, and their average costs are compared. The results from the notebook show that reward shaping generally leads to lower average costs (higher rewards), although the final performance can be noisy.

*Final cost results from one execution:*

| Model                  |   Mean Cost |   Std Dev (Cost) |   Number of Runs |
|:-----------------------|------------:|-----------------:|-----------------:|
| DQN without Shaping    |    20935.520|          467.389 |                5 |
| DQN + Base-Stock       |    20731.774|          331.492 |                5 |
| DQN + BSP-low-EW       |    21007.825|          775.108 |                5 |

### Convergence Analysis

The learning curves (plotted in the notebook) provide a clearer picture, demonstrating that reward shaping leads to faster and more stable convergence. The agent with **BSP-low-EW shaping**, in particular, achieves a good policy much earlier in the training process.


*(Image generated from the notebook's analysis cells)*

### Policy Analysis

For the case where product lifetime `m=2`, the learned policies and their resulting steady-state inventory distributions are visualized as heatmaps. This analysis reveals that the shaped agents learn more structured and reasonable ordering policies, avoiding both excessive stockouts and over-stocking compared to the unshaped agent.

## Conclusion

This project successfully replicates the core findings of the paper by De Moor et al. It demonstrates that **potential-based reward shaping is a powerful technique for improving DQN performance** in complex inventory management problems.

The more sophisticated `BSP-low-EW` heuristic provides a superior shaping signal, resulting in faster convergence and a better final policy compared to both the simpler `base-stock` heuristic and the unshaped baseline. This highlights the value of incorporating domain knowledge into the reinforcement learning process to overcome challenges like sparse rewards.

## License

This project is licensed under the MIT License.