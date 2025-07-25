"""
Perishable Inventory Management Environment for DQN with Reward Shaping
Based on the paper by De Moor et al. (2021)

This module provides:
1. PerishableInventoryEnv: Main environment class
2. RewardShapingWrapper: Wrapper for potential-based reward shaping
3. Utility functions for heuristic policies
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy import stats


class PerishableInventoryEnv(gym.Env):
    """
    Perishable Inventory Management Environment
    
    State space: [inventory_age_1, ..., inventory_age_m, pipeline_1, ..., pipeline_(L-1)]
    Action space: Order quantity [0, 1, ..., q_max]
    """
    
    def __init__(self, m=2, L=1, q_max=30, demand_mean=5, demand_std=2, 
                 c_h=1, c_o=3, c_l=5, c_p=7, delivery_policy='FIFO', max_steps=1000):
        super().__init__()
        
        # Environment parameters
        self.m = m  # Product lifetime
        self.L = L  # Delivery lead time
        self.q_max = q_max  # Maximum order quantity
        self.delivery_policy = delivery_policy.upper()
        self.max_steps = max_steps
        
        # Demand parameters
        self.demand_mean = demand_mean
        self.demand_std = demand_std
        
        # Cost parameters
        self.c_h = c_h  # Holding cost per unit
        self.c_o = c_o  # Ordering cost per unit
        self.c_l = c_l  # Lost sales cost per unit
        self.c_p = c_p  # Perishing cost per unit
        
        # State and action spaces
        state_dim = self.m + max(0, self.L - 1)
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.q_max + 1)
        
        # Initialize state
        self.state = None
        self.step_count = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            
        # Initialize state: [inventory_ages, pipeline_orders]
        inventory = np.zeros(self.m, dtype=np.float32)
        pipeline = np.zeros(max(0, self.L - 1), dtype=np.float32)
        self.state = np.concatenate([inventory, pipeline])
        self.step_count = 0
        
        return self.state, {}
    
    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
            
        # Extract current state
        inventory = self.state[:self.m].copy()
        pipeline = self.state[self.m:].copy() if self.L > 1 else np.array([])
        
        # Generate demand
        demand = max(0, int(np.random.normal(self.demand_mean, self.demand_std)))
        
        # Fulfill demand using FIFO or LIFO
        inventory_after_sales, sales = self._fulfill_demand(inventory, demand)
        lost_sales = demand - sales
        
        # Handle perishing (oldest items perish)
        perished = inventory_after_sales[self.m - 1]
        
        # Age inventory (shift right, oldest items are removed)
        aged_inventory = np.roll(inventory_after_sales, 1)
        aged_inventory[0] = 0  # Clear newest age slot
        
        # Handle deliveries
        if self.L == 1:
            # Immediate delivery
            aged_inventory[0] = action
            new_pipeline = np.array([])
        else:
            # Delivery from pipeline
            if len(pipeline) > 0:
                aged_inventory[0] = pipeline[-1]
                new_pipeline = np.concatenate([[action], pipeline[:-1]])
            else:
                aged_inventory[0] = action
                new_pipeline = np.array([])
        
        # Calculate costs
        holding_cost = self.c_h * np.sum(aged_inventory)
        ordering_cost = self.c_o * action
        lost_sales_cost = self.c_l * lost_sales
        perishing_cost = self.c_p * perished
        
        total_cost = holding_cost + ordering_cost + lost_sales_cost + perishing_cost
        reward = -total_cost
        
        # Update state
        self.state = np.concatenate([aged_inventory, new_pipeline])
        self.step_count += 1
        
        # Check if episode is done
        done = self.step_count >= self.max_steps
        
        info = {
            'total_cost': total_cost,
            'holding_cost': holding_cost,
            'ordering_cost': ordering_cost,
            'lost_sales_cost': lost_sales_cost,
            'perishing_cost': perishing_cost,
            'demand': demand,
            'sales': sales,
            'lost_sales': lost_sales,
            'perished': perished
        }
        
        return self.state, reward, done, False, info
    
    def _fulfill_demand(self, inventory, demand):
        """Fulfill demand using FIFO or LIFO policy"""
        inventory_copy = inventory.copy()
        demand_left = demand
        
        if self.delivery_policy == 'FIFO':
            # Fulfill from oldest items first
            for i in range(self.m - 1, -1, -1):
                fulfilled = min(demand_left, inventory_copy[i])
                inventory_copy[i] -= fulfilled
                demand_left -= fulfilled
                if demand_left == 0:
                    break
        else:  # LIFO
            # Fulfill from newest items first
            for i in range(self.m):
                fulfilled = min(demand_left, inventory_copy[i])
                inventory_copy[i] -= fulfilled
                demand_left -= fulfilled
                if demand_left == 0:
                    break
        
        sales = demand - demand_left
        return inventory_copy, sales


def get_base_stock_level(demand_mean, demand_std, L, service_level=0.95):
    """Calculate base stock level for base-stock policy"""
    # Safety stock calculation
    z_score = stats.norm.ppf(service_level)
    safety_stock = z_score * demand_std * np.sqrt(L + 1)
    base_stock = (L + 1) * demand_mean + safety_stock
    return max(0, int(round(base_stock)))


def get_base_stock_action(state, base_stock_level, m):
    """Get action from base-stock policy"""
    inventory_position = np.sum(state)
    order_quantity = max(0, base_stock_level - inventory_position)
    return int(order_quantity)


def calculate_estimated_waste(inventory, demand_mean, L, m):
    """Calculate estimated waste during lead time"""
    if L <= 0:
        return 0
    
    expected_inventory = inventory.copy()
    total_waste = 0
    
    for _ in range(L):
        # Simulate demand fulfillment (FIFO)
        demand_left = demand_mean
        for i in range(m - 1, -1, -1):
            fulfilled = min(demand_left, expected_inventory[i])
            expected_inventory[i] -= fulfilled
            demand_left -= fulfilled
        
        # Add waste from oldest items
        waste = expected_inventory[m - 1]
        total_waste += waste
        
        # Age inventory
        expected_inventory = np.roll(expected_inventory, 1)
        expected_inventory[0] = 0
    
    return total_waste


def get_bsp_low_ew_action(state, demand_mean, m, L, S1=None, S2=None, b=None, alpha=0.8):
    """Get action from BSP-low-EW policy"""
    inventory = state[:m]
    inventory_position = np.sum(state)
    
    # Default parameters if not provided
    if S1 is None:
        S1 = (L + 1) * demand_mean + 5
    if S2 is None:
        S2 = (L + 1) * demand_mean + 2
    if b is None:
        b = demand_mean * (L + 1) * 0.7
    
    # Calculate estimated waste
    ewt = calculate_estimated_waste(inventory, demand_mean, L, m)
    
    # Apply BSP-low-EW logic
    if inventory_position < b:
        order_quantity = max(0, S1 - alpha * inventory_position + ewt)
    else:
        order_quantity = max(0, S2 - inventory_position + ewt)
    
    return int(order_quantity)


class RewardShapingWrapper(gym.Wrapper):
    """Wrapper to apply potential-based reward shaping"""
    
    def __init__(self, env, shaping_type='none', gamma=0.99, k=1.0):
        super().__init__(env)
        self.shaping_type = shaping_type
        self.gamma = gamma
        self.k = k
        self.last_potential = 0
        
        # Calculate base stock level for base-stock shaping
        if shaping_type == 'base_stock':
            self.base_stock_level = get_base_stock_level(
                env.demand_mean, env.demand_std, env.L
            )
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_potential = self._calculate_potential(obs)
        return obs, info
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Calculate shaping reward
        current_potential = self._calculate_potential(obs)
        shaping_reward = self.gamma * current_potential - self.last_potential
        
        # Add shaping reward to original reward
        shaped_reward = reward + shaping_reward
        
        # Update for next step
        self.last_potential = current_potential
        
        # Add shaping info
        info['original_reward'] = reward
        info['shaping_reward'] = shaping_reward
        info['shaped_reward'] = shaped_reward
        
        return obs, shaped_reward, done, truncated, info
    
    def _calculate_potential(self, state):
        """Calculate potential function value"""
        if self.shaping_type == 'none':
            return 0
        
        elif self.shaping_type == 'base_stock':
            # Potential based on deviation from base-stock action
            target_action = get_base_stock_action(state, self.base_stock_level, self.env.m)
            inventory_position = np.sum(state)
            deviation = abs(inventory_position - self.base_stock_level)
            return -self.k * deviation
        
        elif self.shaping_type == 'bsp_low_ew':
            # Potential based on BSP-low-EW policy
            target_action = get_bsp_low_ew_action(
                state, self.env.demand_mean, self.env.m, self.env.L
            )
            inventory_position = np.sum(state)
            # Use a more sophisticated potential based on inventory position
            optimal_position = (self.env.L + 1) * self.env.demand_mean
            deviation = abs(inventory_position - optimal_position)
            return -self.k * deviation
        
        else:
            raise ValueError(f"Unknown shaping type: {self.shaping_type}")


if __name__ == "__main__":
    # Test the environment
    print("Testing Perishable Inventory Environment...")
    
    env = PerishableInventoryEnv(m=2, L=1, q_max=10, demand_mean=3, demand_std=1)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test basic functionality
    obs, info = env.reset(seed=42)
    print(f"Initial state: {obs}")
    
    for step in range(3):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {step + 1}: Action={action}, Reward={reward:.2f}, State={obs}")
        print(f"  Costs - Total: {info['total_cost']:.2f}, Demand: {info['demand']}")
    
    # Test reward shaping
    print("\nTesting Reward Shaping...")
    shaped_env = RewardShapingWrapper(env, shaping_type='base_stock')
    obs, info = shaped_env.reset(seed=42)
    
    for step in range(2):
        action = shaped_env.action_space.sample()
        obs, reward, done, truncated, info = shaped_env.step(action)
        print(f"Step {step + 1}: Original={info['original_reward']:.2f}, "
              f"Shaping={info['shaping_reward']:.2f}, Total={reward:.2f}")
    
    print("Environment test completed successfully!")
