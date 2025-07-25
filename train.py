import os
import argparse
import time
import numpy as np
import pandas as pd
import torch as th

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from perishable_env import PerishableGymEnv
from reward_shaping_wrapper import RewardShapingWrapper
from common import get_base_stock_level

class EvalCallback(BaseCallback):
    """
    A custom callback for evaluating and logging the agent's performance.
    """
    def __init__(self, eval_env, eval_freq=5000, n_eval_episodes=10, log_path=None, verbose=1):
        super(EvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_path = log_path
        self.eval_timesteps = []
        self.mean_costs = []
        self.std_costs = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            episode_costs = []
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done, total_cost = False, 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, done, info = self.eval_env.step(action)
                    total_cost += info.get('cost', -info.get('original_reward', 0))
                episode_costs.append(total_cost)

            mean_cost = np.mean(episode_costs)
            std_cost = np.std(episode_costs)
            
            self.eval_timesteps.append(self.num_timesteps)
            self.mean_costs.append(mean_cost)
            self.std_costs.append(std_cost)
            
            if self.verbose > 0:
                print(f"Step {self.num_timesteps}: Mean cost: {mean_cost:.2f} +/- {std_cost:.2f}")
                
            if self.log_path:
                df = pd.DataFrame({
                    'timesteps': self.eval_timesteps,
                    'mean_cost': self.mean_costs,
                    'std_cost': self.std_costs
                })
                df.to_csv(self.log_path, index=False)

        return True

def main(args):
    # --- Create directories ---
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # --- Environment Setup ---
    env_params = {
        'm': args.m, 'L': args.L, 'demand_mean': args.demand_mean, 'demand_cov': args.demand_cov,
        'cp': args.cp, 'cl': args.cl, 'ch': args.ch, 'co': args.co,
        'issuing_policy': args.issuing_policy
    }
    env = PerishableGymEnv(**env_params)
    
    # --- Reward Shaping Setup ---
    if args.shaping_type != 'none':
        demand_std = args.demand_mean * args.demand_cov
        if args.shaping_type == 'base_stock':
            target_level = get_base_stock_level(args.demand_mean, demand_std, args.L)
        elif args.shaping_type == 'bsp_low_ew':
            # Note: BSP-low-EW has its own dynamic targets, but for a simple state potential,
            # we'll use a heuristic target level. A more advanced implementation might
            # pass the teacher's action. For this project, we use a fixed target.
            target_level = get_base_stock_level(args.demand_mean, demand_std, args.L, service_level=0.98)
        else:
            raise ValueError(f"Unknown shaping type: {args.shaping_type}")

        shaping_params = {'k': args.shaping_k, 'target_level': target_level}
        env = RewardShapingWrapper(env, shaping_params=shaping_params, gamma=args.gamma)

    env = DummyVecEnv([lambda: env])
    
    # --- Callback and Logging ---
    log_file_name = f"eval_log_{args.shaping_type}_seed{args.seed}.csv"
    log_path = os.path.join(args.log_dir, log_file_name)
    eval_callback = EvalCallback(env, eval_freq=args.eval_freq, log_path=log_path)

    # --- Model Definition ---
    policy_kwargs = dict(net_arch=[int(x) for x in args.net_arch.split(',')])
    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        train_freq=(1, "episode"),
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        tensorboard_log=f"./{args.log_dir}/",
        seed=args.seed,
        verbose=1,
    )

    # --- Training ---
    print(f"Starting training for {args.shaping_type} with seed {args.seed}...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
        tb_log_name=f"dqn_{args.shaping_type}_seed{args.seed}"
    )

    # --- Save Model ---
    model_path = os.path.join(args.model_dir, f"dqn_{args.shaping_type}_seed{args.seed}.zip")
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Env params
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--demand_mean", type=float, default=4)
    parser.add_argument("--demand_cov", type=float, default=0.5)
    parser.add_argument("--cp", type=int, default=7)
    parser.add_argument("--cl", type=int, default=5)
    parser.add_argument("--ch", type=int, default=1)
    parser.add_argument("--co", type=int, default=3)
    parser.add_argument("--issuing_policy", type=str, default="LIFO", choices=["FIFO", "LIFO"])
    
    # Training params
    parser.add_argument("--total_timesteps", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--net_arch", type=str, default="32,32")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--buffer_size", type=int, default=20000)
    parser.add_argument("--learning_starts", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target_update_interval", type=int, default=20) # In episodes
    parser.add_argument("--exploration_fraction", type=float, default=0.9) # Corresponds to epsilon decay
    parser.add_argument("--exploration_final_eps", type=float, default=0.01)
    
    # Shaping params
    parser.add_argument("--shaping_type", type=str, default="none", choices=["none", "base_stock", "bsp_low_ew"])
    parser.add_argument("--shaping_k", type=float, default=50.0)

    # Logging and saving
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--eval_freq", type=int, default=5000)

    args = parser.parse_args()
    main(args) 