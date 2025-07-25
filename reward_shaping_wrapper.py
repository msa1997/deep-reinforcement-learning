import gym
import numpy as np

class RewardShapingWrapper(gym.Wrapper):
    """
    A Gym wrapper to apply potential-based reward shaping.
    The shaping reward is F(s,a,s') = gamma * Φ(s') - Φ(s).
    The potential function Φ(s) is defined based on the deviation from a
    target inventory level.
    """
    def __init__(self, env, shaping_params, gamma):
        super(RewardShapingWrapper, self).__init__(env)
        self.shaping_params = shaping_params
        self.gamma = gamma
        self._state_before_step = None

    def _calculate_potential(self, state):
        """
        Calculates the potential of a given state.
        Φ(s) = -k * |inventory_position - target_level|
        """
        k = self.shaping_params.get('k', 1.0)
        target_level = self.shaping_params.get('target_level')

        if target_level is None:
            raise ValueError("shaping_params must contain a 'target_level' for the potential function.")

        # Inventory position is the sum of on-hand inventory
        inventory_position = np.sum(state[:self.env.unwrapped.m])
        deviation = abs(inventory_position - target_level)
        return -k * deviation

    def step(self, action):
        """
        - Calculates the potential of the current state, Φ(s).
        - Takes a step in the environment to get the next state, s'.
        - Calculates the potential of the next state, Φ(s').
        - Computes the shaping reward and adds it to the environment's reward.
        """
        # Potential of the state *before* the action
        current_potential = self._calculate_potential(self._state_before_step)

        # Take step
        next_state, reward, done, info = self.env.step(action)
        self._state_before_step = next_state.copy()

        # Potential of the state *after* the action
        next_potential = self._calculate_potential(next_state)

        # Shaping reward
        shaping_reward = self.gamma * next_potential - current_potential
        shaped_reward = reward + shaping_reward

        info['original_reward'] = reward
        info['shaping_reward'] = shaping_reward

        return next_state, shaped_reward, done, info

    def reset(self, **kwargs):
        """Reset the environment and store the initial state."""
        obs = self.env.reset(**kwargs)
        self._state_before_step = obs.copy()
        return obs 