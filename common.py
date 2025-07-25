import numpy as np
from scipy.stats import gamma

def get_base_stock_policy_action(state, base_stock_level, m, L):
    """
    Calculates the action for a base-stock policy.
    Orders up to the base_stock_level.
    """
    inventory_position = np.sum(state)
    order_quantity = max(0, base_stock_level - inventory_position)
    return int(round(order_quantity))

def get_bsp_low_ew_policy_action(state, S1, S2, b, alpha, demand_mean, m, L):
    """
    Calculates the action for the BSP-low-EW policy from the paper.
    This policy uses two order-up-to levels depending on the inventory position.
    """
    inventory_position = np.sum(state[:m])
    
    # Calculate estimated waste during lead time
    ewt = calculate_estimated_waste(state[:m], demand_mean, L, m)
    
    if inventory_position < b:
        order_quantity = max(0, S1 - alpha * inventory_position + ewt)
    else:
        order_quantity = max(0, S2 - inventory_position + ewt)
        
    return int(round(order_quantity))

def calculate_estimated_waste(inventory, demand_mean, L, m):
    """
    Estimates the waste that will occur during the lead time.
    This is a simplified version assuming mean demand in each period.
    """
    if L == 0:
        return 0

    expected_inventory = inventory.copy()
    total_waste = 0

    for _ in range(L):
        # Fulfill demand from oldest items first for waste calculation
        demand_left = demand_mean
        inventory_after_sales = expected_inventory.copy()
        for i in range(m - 1, -1, -1):
            fulfilled = min(demand_left, inventory_after_sales[i])
            inventory_after_sales[i] -= fulfilled
            demand_left -= fulfilled
            
        waste = inventory_after_sales[m - 1]
        total_waste += waste
        
        # Age the inventory
        expected_inventory = np.roll(inventory_after_sales, 1)
        expected_inventory[0] = 0  # New arrivals are not part of this calculation

    return total_waste

def get_base_stock_level(demand_mean, demand_std, L, service_level=0.95):
    """
    Calculates a simple base-stock level.
    """
    # Safety stock based on service level (z-score for normal distribution)
    # This is an approximation.
    z = gamma.ppf(service_level, a=L+1) # Approximation
    safety_stock = z * demand_std * np.sqrt(L + 1)
    
    base_stock = (L + 1) * demand_mean + safety_stock
    return int(round(base_stock)) 