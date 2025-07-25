"""
Training script for DQN with Reward Shaping in Perishable Inventory Management
This script can be run independently to train and evaluate models.
"""

import os
import time
import random
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from perishable_inventory_env_augment import PerishableInventoryEnv, RewardShapingWrapper


def set_seeds(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def create_dqn_model(env, learning_rate=0.001, buffer_size=50000, 
                     learning_starts=1000, target_update_interval=1000,
                     exploration_fraction=0.1, exploration_final_eps=0.02,
                     seed=None, verbose=0):
    """Create DQN model with specified parameters"""
    
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=exploration_final_eps,
        max_grad_norm=10,
        tensorboard_log=None,
        policy_kwargs=dict(net_arch=[64, 64]),  # Two hidden layers with 64 neurons each
        verbose=verbose,
        seed=seed
    )
    
    return model


def train_and_evaluate(shaping_type, seed, env_params, total_timesteps=50000, 
                      eval_freq=5000, n_eval_episodes=10, verbose=True):
    """Train a single model and return evaluation results"""
    
    if verbose:
        print(f"Training {shaping_type} with seed {seed}...")
    
    # Set seeds
    set_seeds(seed)
    
    # Create environment
    base_env = PerishableInventoryEnv(**env_params)
    
    if shaping_type == 'none':
        env = base_env
    else:
        env = RewardShapingWrapper(base_env, shaping_type=shaping_type)
    
    # Create model
    model = create_dqn_model(env, seed=seed, verbose=0)
    
    # Training with periodic evaluation
    eval_timesteps = []
    eval_mean_rewards = []
    eval_std_rewards = []
    
    for timestep in range(0, total_timesteps + 1, eval_freq):
        if timestep > 0:
            model.learn(total_timesteps=eval_freq, reset_num_timesteps=False)
        
        # Evaluate current policy
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=n_eval_episodes, deterministic=True
        )
        
        eval_timesteps.append(timestep)
        eval_mean_rewards.append(mean_reward)
        eval_std_rewards.append(std_reward)
        
        if verbose:
            print(f"  Timestep {timestep}: Mean reward = {mean_reward:.2f} ± {std_reward:.2f}")
    
    return {
        'timesteps': eval_timesteps,
        'mean_rewards': eval_mean_rewards,
        'std_rewards': eval_std_rewards,
        'final_mean': eval_mean_rewards[-1],
        'final_std': eval_std_rewards[-1],
        'model': model
    }


def run_experiment(shaping_types=['none', 'base_stock', 'bsp_low_ew'], 
                   seeds=[42, 123, 456], 
                   env_params=None,
                   total_timesteps=50000,
                   eval_freq=5000,
                   n_eval_episodes=10):
    """Run complete experiment for multiple shaping types and seeds"""
    
    if env_params is None:
        env_params = {
            'm': 2,
            'L': 1, 
            'q_max': 30,
            'demand_mean': 5,
            'demand_std': 2,
            'c_h': 1,
            'c_o': 3, 
            'c_l': 5,
            'c_p': 7,
            'delivery_policy': 'FIFO'
        }
    
    print("Starting DQN training experiment...")
    print(f"Shaping types: {shaping_types}")
    print(f"Seeds: {seeds}")
    print(f"Environment parameters: {env_params}")
    print(f"Total timesteps per run: {total_timesteps}")
    print("-" * 60)
    
    all_results = {}
    
    for shaping_type in shaping_types:
        print(f"\n=== {shaping_type.upper()} SHAPING ===")
        
        results = {}
        for i, seed in enumerate(seeds):
            start_time = time.time()
            result = train_and_evaluate(
                shaping_type, seed, env_params, total_timesteps, 
                eval_freq, n_eval_episodes, verbose=True
            )
            training_time = time.time() - start_time
            result['training_time'] = training_time
            results[seed] = result
            
            print(f"  Seed {seed} completed in {training_time:.1f}s")
        
        all_results[shaping_type] = results
        
        # Calculate summary statistics
        final_rewards = [result['final_mean'] for result in results.values()]
        final_costs = [-reward for reward in final_rewards]
        mean_cost = np.mean(final_costs)
        std_cost = np.std(final_costs)
        
        print(f"  Summary: Mean cost = {mean_cost:.2f} ± {std_cost:.2f}")
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETED!")
    print("=" * 60)
    
    # Print final comparison
    print("\nFINAL COMPARISON:")
    for shaping_type, results in all_results.items():
        final_rewards = [result['final_mean'] for result in results.values()]
        final_costs = [-reward for reward in final_rewards]
        mean_cost = np.mean(final_costs)
        std_cost = np.std(final_costs)
        print(f"{shaping_type:12}: {mean_cost:.2f} ± {std_cost:.2f}")
    
    return all_results


def quick_test():
    """Quick test to verify everything works"""
    print("Running quick test...")
    
    env_params = {
        'm': 2, 'L': 1, 'q_max': 10, 'demand_mean': 3, 'demand_std': 1,
        'c_h': 1, 'c_o': 3, 'c_l': 5, 'c_p': 7, 'delivery_policy': 'FIFO'
    }
    
    results = run_experiment(
        shaping_types=['none', 'base_stock'], 
        seeds=[42, 123], 
        env_params=env_params,
        total_timesteps=10000,
        eval_freq=2500,
        n_eval_episodes=5
    )
    
    print("Quick test completed successfully!")
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DQN with reward shaping')
    parser.add_argument('--quick-test', action='store_true', 
                       help='Run quick test with reduced parameters')
    parser.add_argument('--timesteps', type=int, default=50000,
                       help='Total timesteps for training')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456],
                       help='Random seeds to use')
    parser.add_argument('--shaping', nargs='+', default=['none', 'base_stock', 'bsp_low_ew'],
                       help='Shaping types to test')
    
    args = parser.parse_args()
    
    if args.quick_test:
        quick_test()
    else:
        run_experiment(
            shaping_types=args.shaping,
            seeds=args.seeds,
            total_timesteps=args.timesteps
        )
