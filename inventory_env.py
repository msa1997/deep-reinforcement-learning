import gymnasium as gym
import numpy as np
from gymnasium import spaces

class InventoryEnv(gym.Env):
    """
    A simple inventory management environment for reinforcement learning.
    
    The agent manages inventory levels to minimize costs while meeting demand.
    """
    
    def __init__(self, max_inventory=100, max_order=50, demand_mean=10, demand_std=3):
        super(InventoryEnv, self).__init__()
        
        self.max_inventory = max_inventory
        self.max_order = max_order
        self.demand_mean = demand_mean
        self.demand_std = demand_std
        
        # Define action space (order quantity)
        self.action_space = spaces.Discrete(max_order + 1)
        
        # Define observation space (current inventory level)
        self.observation_space = spaces.Box(
            low=0, high=max_inventory, shape=(1,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.inventory = np.random.randint(0, self.max_inventory // 2)
        self.time_step = 0
        return np.array([self.inventory], dtype=np.float32), {}
    
    def step(self, action):
        # Ensure action is within bounds
        order_quantity = min(action, self.max_order)
        
        # Add order to inventory
        self.inventory += order_quantity
        
        # Generate random demand
        demand = max(0, int(np.random.normal(self.demand_mean, self.demand_std)))
        
        # Fulfill demand
        fulfilled = min(self.inventory, demand)
        self.inventory -= fulfilled
        
        # Calculate costs
        holding_cost = 0.1 * self.inventory  # Cost per unit held
        ordering_cost = 0.5 * order_quantity  # Cost per unit ordered
        shortage_cost = 2.0 * (demand - fulfilled)  # Cost per unit of unmet demand
        
        total_cost = holding_cost + ordering_cost + shortage_cost
        reward = -total_cost  # Negative cost as reward
        
        # Check if episode is done (after 100 time steps)
        self.time_step += 1
        done = self.time_step >= 100
        
        return (
            np.array([self.inventory], dtype=np.float32),
            reward,
            done,
            False,
            {
                'demand': demand,
                'fulfilled': fulfilled,
                'order_quantity': order_quantity,
                'total_cost': total_cost
            }
        )
    
    def render(self):
        print(f"Inventory: {self.inventory}, Time Step: {self.time_step}")
    
    def close(self):
        pass


class PerishableInventoryEnv(gym.Env):
    """
    A perishable inventory management environment for reinforcement learning.
    
    The agent manages inventory levels considering product expiration dates.
    """
    
    def __init__(self, max_inventory=100, max_order=50, shelf_life=7, demand_mean=10, demand_std=3):
        super(PerishableInventoryEnv, self).__init__()
        
        self.max_inventory = max_inventory
        self.max_order = max_order
        self.shelf_life = shelf_life
        self.demand_mean = demand_mean
        self.demand_std = demand_std
        
        # Define action space (order quantity)
        self.action_space = spaces.Discrete(max_order + 1)
        
        # Define observation space (inventory levels by age + current inventory)
        self.observation_space = spaces.Box(
            low=0, high=max_inventory, shape=(shelf_life + 1,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        # Initialize inventory by age (0 = fresh, shelf_life-1 = oldest)
        self.inventory_by_age = np.zeros(self.shelf_life, dtype=np.int32)
        self.time_step = 0
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get current observation including inventory by age and total inventory."""
        obs = np.concatenate([
            self.inventory_by_age.astype(np.float32),
            [np.sum(self.inventory_by_age)]
        ])
        return obs
    
    def step(self, action):
        # Ensure action is within bounds
        order_quantity = min(action, self.max_order)
        
        # Add new order to fresh inventory (age 0)
        self.inventory_by_age[0] += order_quantity
        
        # Generate random demand
        demand = max(0, int(np.random.normal(self.demand_mean, self.demand_std)))
        
        # Fulfill demand using FIFO (First In, First Out) - oldest items first
        remaining_demand = demand
        fulfilled = 0
        
        for age in range(self.shelf_life - 1, -1, -1):  # Start from oldest
            if remaining_demand > 0 and self.inventory_by_age[age] > 0:
                used = min(remaining_demand, self.inventory_by_age[age])
                self.inventory_by_age[age] -= used
                fulfilled += used
                remaining_demand -= used
        
        # Age inventory (move items to next age group)
        expired = self.inventory_by_age[-1]  # Items that expired
        self.inventory_by_age[1:] = self.inventory_by_age[:-1]  # Age all items
        self.inventory_by_age[0] = 0  # Reset fresh inventory (new orders will be added next step)
        
        # Calculate costs
        total_inventory = np.sum(self.inventory_by_age)
        holding_cost = 0.1 * total_inventory  # Cost per unit held
        ordering_cost = 0.5 * order_quantity  # Cost per unit ordered
        shortage_cost = 2.0 * (demand - fulfilled)  # Cost per unit of unmet demand
        expiration_cost = 1.0 * expired  # Cost per unit expired
        
        total_cost = holding_cost + ordering_cost + shortage_cost + expiration_cost
        reward = -total_cost  # Negative cost as reward
        
        # Check if episode is done (after 100 time steps)
        self.time_step += 1
        done = self.time_step >= 100
        
        return (
            self._get_observation(),
            reward,
            done,
            False,
            {
                'demand': demand,
                'fulfilled': fulfilled,
                'order_quantity': order_quantity,
                'expired': expired,
                'total_cost': total_cost,
                'inventory_by_age': self.inventory_by_age.copy()
            }
        )
    
    def render(self):
        print(f"Inventory by age: {self.inventory_by_age}, Total: {np.sum(self.inventory_by_age)}, Time Step: {self.time_step}")
    
    def close(self):
        pass 