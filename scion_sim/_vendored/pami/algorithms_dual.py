import numpy as np
from collections import deque

class DualAlgorithmDDE:
    def __init__(self, network, params, dt=0.0001):
        self.network = network
        self.params = params
        self.dt = dt
        self.p = params['p']
        self.q = params['q'] 
        self.alpha = params['alpha']
        
        # --- Pre-calculate delay steps ---
        for path in self.network['path_list']:
            path.delay_steps_Tr = int(round(path.rtt / dt))
            path.delay_steps_Tjr = {}
            path.delay_steps_Trj = {}
            for res in path.resources:
                # Delays from Table I
                if res.id == 'j1': delay = 0.01
                elif res.id == 'j2': delay = 0.1
                elif res.id == 'j3': delay = 1.0
                path.delay_steps_Tjr[res.id] = int(round(delay / dt))
                path.delay_steps_Trj[res.id] = int(round(delay / dt))

    def run(self, warmup_duration, experiment_duration):
        steps_warmup = int(warmup_duration / self.dt)
        steps_exp = int(experiment_duration / self.dt)
        
        # --- SMART INITIALIZATION ---
        # Instead of guessing 5.0 and waiting, we calculate the exact equilibrium.
        # 1. Flows (x) must be 5.0 (Symmetry + Capacity 10)
        # 2. Source State (y) = (Sum x^q)^(1/q) = (2 * 5^0.875)^(1.14) approx 11.026
        # 3. Price (mu) calculated from Eq 8 approx 0.01
        
        x_init = 5.0
        y_init = 11.026  # precise value: (2 * 5**(7/8))**(8/7)
        mu_init = 0.01   # approx value to sustain x=5
        
        current_y = {s.id: y_init for s in self.network['source_list']}
        current_mu = {j.id: mu_init for j in self.network['resource_list']}
        
        # Initialize history buffers with these stable values
        maxlen = 25000
        history_x = {p.id: deque([x_init]*maxlen, maxlen=maxlen) for p in self.network['path_list']}
        history_mu = {j.id: deque([mu_init]*maxlen, maxlen=maxlen) for j in self.network['resource_list']}
        
        results = {'time': [], 'x': {p.id: [] for p in self.network['path_list']}}
        
        print(f"Simulation configured: {warmup_duration}s warmup + {experiment_duration}s experiment")
        
        total_steps = steps_warmup + steps_exp
        for step in range(total_steps):
            t = (step * self.dt) - warmup_duration
            
            # Failure Event at t=0
            failed_routes = []
            if t >= 0:
                self.network['resources']['j2'].capacity = 0
                failed_routes = ['r2', 'r3']
            
            # 1. Update Flows x(t)
            current_x = {}
            for p in self.network['path_list']:
                if p.id in failed_routes:
                    current_x[p.id] = 0.0
                else:
                    lambda_r = 0.0
                    for res in p.resources:
                        delay_idx = p.delay_steps_Tjr[res.id]
                        lambda_r += history_mu[res.id][-(delay_idx+1)]
                    
                    lambda_r = max(lambda_r, 1e-9)
                    
                    # Eq (8)
                    w_s = self.params['w_s'][p.source_id]
                    y_s = current_y[p.source_id]
                    
                    term1 = lambda_r ** -(self.p + 1)
                    term2 = w_s ** (self.p + 1)
                    term3 = y_s ** (1.0 - (self.alpha * (self.p + 1)))
                    
                    current_x[p.id] = term1 * term2 * term3
            
            for pid, val in current_x.items():
                history_x[pid].append(val)

            # 2. Update Source State y(t)
            dy = {}
            for s in self.network['source_list']:
                sum_xq = 0.0
                s_routes = [p for p in self.network['path_list'] if p.source_id == s.id]
                for p in s_routes:
                    x_delayed = history_x[p.id][-(p.delay_steps_Tr+1)]
                    sum_xq += x_delayed ** self.q
                
                y_val = current_y[s.id]
                k_s = self.params['kappa_s'][s.id]
                dy[s.id] = k_s * (y_val ** (1.0/(self.p + 1))) * (sum_xq - (y_val ** self.q))

            # 3. Update Prices mu(t)
            dmu = {}
            for res in self.network['resource_list']:
                z_j = 0.0
                res_routes = [p for p in self.network['path_list'] if res in p.resources]
                for p in res_routes:
                    x_delayed = history_x[p.id][-(p.delay_steps_Trj[res.id]+1)]
                    z_j += x_delayed
                
                mu_val = current_mu[res.id]
                k_j = self.params['kappa_j'][res.id]
                dmu[res.id] = k_j * mu_val * (z_j - res.capacity)

            # 4. Integrate
            for sid in current_y:
                current_y[sid] = max(current_y[sid] + dy[sid] * self.dt, 1e-6)
            for rid in current_mu:
                current_mu[rid] = max(current_mu[rid] + dmu[rid] * self.dt, 1e-9)
                history_mu[rid].append(current_mu[rid])
            
            # Save data (downsampled)
            if t >= -10 and step % 100 == 0:
                results['time'].append(t)
                for pid in current_x:
                    results['x'][pid].append(current_x[pid])
        
        return results