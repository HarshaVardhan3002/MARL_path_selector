# algorithms_pan.py
import numpy as np

def pan_primal_equations(t, state, network, params):
    """
    An adapted version of the primal algorithm for the PAN simulation.
    This version knows about the scheduler and probing traffic.
    """
    # --- Unpack parameters and network components ---
    a_bar, b_r, beta_j = params['a_bar'], params['b_r'], params['beta_j']
    
    # IMPORTANT: We now only consider the paths the scheduler has made active
    active_paths = network['active_paths']
    resources = network['resources']
    scheduler = network['scheduler']

    # The state vector corresponds ONLY to the flow rates of the active paths
    flow_rates = {p.id: state[i] for i, p in enumerate(active_paths)}

    # --- Calculate intermediate values ---

    # Calculate z_j: total load on each resource
    z = {res_id: 0 for res_id in resources}
    # 1. Add data traffic from ACTIVE paths
    for p in active_paths:
        if flow_rates[p.id] > 0:
            for res in p.resources:
                z[res.id] += flow_rates[p.id]
    
    # 2. Add the "PROBE STORM" traffic from the scheduler
    probing_load_per_resource = scheduler.get_probing_load() / len(resources)
    for res_id in z:
        z[res_id] += probing_load_per_resource

    # Calculate penalty_j for each resource (same as before)
    penalty_j = {
        res_id: (z[res_id] / res.capacity) ** beta_j
        for res_id, res in resources.items()
    }

    # Calculate lambda_r for each ACTIVE path
    lambda_r = {}
    for p in active_paths:
        prod_term = np.prod([(1 - penalty_j[res.id]) for res in p.resources])
        lambda_r[p.id] = 1 - prod_term
    
    # Calculate y_s: total flow from the source (from active paths only)
    y_s = sum(flow_rates.values())

    # --- Calculate the derivatives dx_r/dt for ACTIVE paths ---
    derivatives = []
    for i, p in enumerate(active_paths):
        x_r = state[i]
        
        inner_term = a_bar * (1 - lambda_r[p.id]) - b_r * y_s * lambda_r[p.id]
        dx_dt = (x_r / p.rtt) * inner_term
        
        if x_r <= 0 and dx_dt < 0:
            derivatives.append(0)
        else:
            derivatives.append(dx_dt)

    return derivatives