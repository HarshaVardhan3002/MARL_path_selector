# algorithms.py
import numpy as np

def primal_model_equations(t, state, network, params):
    """
    Defines the system of differential equations for the primal algorithm.
    """
    # Unpack network components and parameters for easier access
    paths = network['path_list']
    resources = network['resources']
    a_bar = params['a_bar']
    b_r = params['b_r']
    beta_j = params['beta_j']

    # 'state' is the array of current flow rates, x_r(t)
    # We create a dictionary for easy lookup of a path's flow rate
    flow_rates = {p.id: state[i] for i, p in enumerate(paths)}

    # --- Calculate intermediate values from the paper ---

    # Calculate z_j for each resource: total flow through it. (Eq. 11, adapted)
    z = {res_id: 0 for res_id in resources}
    for p in paths:
        if flow_rates[p.id] > 0:
            for res in p.resources:
                z[res.id] += flow_rates[p.id]

    # Calculate mu_j for each resource: the penalty function. (Eq. 4 & 5)
    # Note: The paper uses mu_j for this, but also for price in the dual model.
    # We'll call it penalty_j to avoid confusion.
    penalty_j = {}
    for res_id, res in resources.items():
        if res.capacity > 0:
            penalty_j[res_id] = (z[res_id] / res.capacity) ** beta_j
        else:
            penalty_j[res_id] = 1.0 # If capacity is 0, it's fully congested

    # Calculate lambda_r for each path: the path congestion signal. (Eq. 2)
    lambda_r = {}
    for p in paths:
        # Product over (1 - penalty) for all resources on the path
        prod_term = 1.0
        for res in p.resources:
            prod_term *= (1 - penalty_j[res.id])
        lambda_r[p.id] = 1 - prod_term
    
    # --- Calculate the derivatives dx_r/dt ---
    
    # Calculate y_s for each source: total flow from that source. (Eq. 3)
    y_s = {src_id: 0 for src_id in network['sources']}
    for p in paths:
        y_s[p.source_id] += flow_rates[p.id]

    # Finally, calculate the derivative for each flow using Eq. (1)
    derivatives = []
    for i, p in enumerate(paths):
        x_r = state[i]
        if x_r <= 0: # Ensure flow does not go negative
            derivatives.append(0)
            continue
        
        # This is the term inside the max(0, ...) part of Eq. (1)
        # Note: The paper has a typo y_s(r)(t)lambda_r(t), it should be y_s(r)(t)
        inner_term = a_bar * (1 - lambda_r[p.id]) - b_r * y_s[p.source_id] * lambda_r[p.id]

        # The ( ... )_x_r^+ notation means the derivative is 0 if x_r is 0
        # and the term in parenthesis is negative. Our solver handles this.
        dx_dt = (x_r / p.rtt) * inner_term
        
        # If the term is negative, the flow should decrease.
        # However, we must prevent the flow from becoming negative.
        # Check if any resource on the path has failed (capacity is 0)
        path_failed = any(res.capacity == 0 for res in p.resources)

        if path_failed:
            derivatives.append(-x_r / 0.01) # Force flow to decay to zero quickly
        elif x_r <= 0 and dx_dt < 0:
            derivatives.append(0) # Prevent flow from becoming negative
        else:
            derivatives.append(dx_dt)

    return derivatives

# (The dual_model_equations function remains as a placeholder for now)
def dual_model_equations(t, state, network, params):
    return np.zeros_like(state)