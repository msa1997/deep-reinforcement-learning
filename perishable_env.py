import gym
import numpy as np
from gym import spaces

class PerishableGymEnv(gym.Env):
    """
    A perishable inventory management environment for OpenAI Gym, based on the paper
    "Reward shaping to improve the performance of deep reinforcement learning in
    perishable inventory management" by De Moor et al. (2021).
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, m, L, demand_mean, demand_cov, q_max=10, ch=1, co=3, cl=5, cp=7,
                 issuing_policy='FIFO'):
        super(PerishableGymEnv, self).__init__()

        # Environment parameters
        self.m = m  # product lifetime
        self.L = L  # delivery lead time
        self.q_max = q_max
        self.issuing_policy = issuing_policy.upper()

        # Demand parameters
        self.demand_mean = demand_mean
        self.demand_cov = demand_cov
        gamma_shape = 1 / (demand_cov ** 2)
        gamma_scale = demand_mean * (demand_cov ** 2)
        self.demand_gamma_params = {'shape': gamma_shape, 'scale': gamma_scale}

        # Costs
        self.ch = ch  # holding cost per unit
        self.co = co  # order cost per unit
        self.cl = cl  # lost sales cost per unit
        self.cp = cp  # perishing cost per unit

        # State space: [inventory_age_1, ..., inventory_age_m, pipeline_order_1, ..., pipeline_order_L-1]
        self.state_dim = self.m + self.L - 1
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        # Action space: discrete order quantities [0, 1, ..., q_max]
        self.action_space = spaces.Discrete(self.q_max + 1)

        self.state = None

    def _get_demand(self):
        """Sample demand from a Gamma distribution."""
        demand = np.random.gamma(**self.demand_gamma_params)
        return int(round(demand))

    def step(self, action):
        """Execute one time step within the environment."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        # Unpack state
        current_inventory = self.state[:self.m].copy()
        pipeline = self.state[self.m:].copy()

        # --- Fulfill demand from current inventory ---
        demand = self._get_demand()
        inventory_after_sales = current_inventory.copy()
        
        if self.issuing_policy == 'FIFO':
            demand_left = demand
            for i in range(self.m - 1, -1, -1):  # Oldest items first
                fulfilled = min(demand_left, inventory_after_sales[i])
                inventory_after_sales[i] -= fulfilled
                demand_left -= fulfilled
        elif self.issuing_policy == 'LIFO':
            demand_left = demand
            for i in range(self.m):  # Youngest items first
                fulfilled = min(demand_left, inventory_after_sales[i])
                inventory_after_sales[i] -= fulfilled
                demand_left -= fulfilled
        else:
            raise ValueError("Invalid issuing policy. Choose 'FIFO' or 'LIFO'.")
        
        sales = np.sum(current_inventory) - np.sum(inventory_after_sales)
        lost_sales = demand - sales

        # --- Inventory aging and perishing ---
        perished_units = inventory_after_sales[self.m - 1]
        
        next_inventory_aged = np.roll(inventory_after_sales, 1)
        next_inventory_aged[0] = 0 # Newest slot is empty after aging
        
        # --- New delivery arrives ---
        if self.L == 1:
            arriving_order = action
        elif self.L > 1:
            arriving_order = pipeline[-1] # order from t-(L-1)
        else: # L=0
            arriving_order = action
            
        next_inventory = next_inventory_aged
        next_inventory[0] = arriving_order

        # --- Update pipeline ---
        next_pipeline = np.zeros(self.L - 1)
        if self.L > 1:
            next_pipeline[0] = action
            next_pipeline[1:] = pipeline[:-1]

        # --- Calculate costs and reward ---
        holding_cost = self.ch * np.sum(inventory_after_sales)
        order_cost = self.co * action
        lost_sales_cost = self.cl * lost_sales
        perishing_cost = self.cp * perished_units
        total_cost = holding_cost + order_cost + lost_sales_cost + perishing_cost
        reward = -total_cost

        # --- Update state and return ---
        self.state = np.concatenate([next_inventory, next_pipeline]).astype(np.float32)
        
        return self.state, reward, False, {"cost": total_cost}

    def reset(self):
        """Reset the state of the environment to an initial state."""
        self.state = np.zeros(self.state_dim, dtype=np.float32)
        return self.state

    def render(self, mode='human', close=False):
        """Render the environment to the screen."""
        print(f'State: {self.state}')
        print(f'Action: {self.action_space.sample()}')
        
if __name__ == '__main__':
    # Example of using the environment
    env = PerishableGymEnv(m=2, L=1, demand_mean=4, demand_cov=0.5)
    obs = env.reset()
    print("Initial observation:", obs)
    
    total_reward = 0
    for _ in range(10):
        action = env.action_space.sample()  # Sample a random action
        obs, reward, done, info = env.step(action)
        total_reward += reward
        print(f"Action: {action}, Obs: {obs}, Reward: {reward}, Cost: {info['cost']}")
        if done:
            break
    
    print("Total reward over 10 steps:", total_reward)
    env.close() 