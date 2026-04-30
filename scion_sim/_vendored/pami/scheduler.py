# scheduler.py
import numpy as np
import random

class FairnessEstimator:
    """
    Estimates Axiom 4 (Fairness) using the Markov Process 
    described in Scherrer et al. (Section 5.2).
    Simulates a population of agents to track variance of cwnd.
    """
    def __init__(self, num_agents=1000, alpha=1, beta=0.7):
        self.N = num_agents
        self.alpha = alpha # Additive Increase
        self.beta = beta   # Multiplicative Decrease
        # Initialize all agents with a small starting window
        self.cwnds = np.ones(num_agents) 

    def step(self, m, r, loss_probability=0.0):
        """
        Updates the distribution of windows for one time step.
        m: migration probability (responsiveness)
        r: reset softness (0=hard reset, 1=no reset)
        loss_probability: probability of loss event in this step
        """
        # Generate random values for all agents at once for performance
        rand_m = np.random.random(self.N)
        rand_loss = np.random.random(self.N)
        
        # Determine states for each agent
        migrating = rand_m < m
        # Loss only affects those NOT migrating (as per paper logic, migration takes precedence or happens concurrently)
        experiencing_loss = (rand_loss < loss_probability) & (~migrating)
        increasing = (~migrating) & (~experiencing_loss)

        # Apply updates
        # 1. Migration: Apply reset softness 'r'
        self.cwnds[migrating] = self.cwnds[migrating] * r
        
        # 2. Loss: Apply multiplicative decrease 'beta'
        self.cwnds[experiencing_loss] = self.cwnds[experiencing_loss] * self.beta
        
        # 3. Increase: Apply additive increase 'alpha'
        self.cwnds[increasing] += self.alpha

    def get_variance(self):
        """Returns the current fairness metric eta (variance)."""
        return np.var(self.cwnds)


class NaiveScheduler:
    """
    A simple but aggressive scheduler that causes instability.
    Constant probing creates the 'Probe Storm'.
    """
    def __init__(self, source_id, all_paths, probe_rate=0.1, active_set_size=2):
        self.source_id = source_id
        self.all_paths = [p for p in all_paths if p.source_id == source_id]
        self.probe_rate = probe_rate
        self.active_set_size = active_set_size
        
        self.all_paths.sort(key=lambda p: p.rtt)
        self.active_paths = self.all_paths[:self.active_set_size]
        
        # Axiom 4: Estimator (Naive = High Static Probe Rate, effectively constant disturbance)
        # We assume high 'm' because it's constantly looking to switch
        self.fairness_estimator = FairnessEstimator()

    def get_active_paths(self):
        return self.active_paths

    def get_probing_load(self):
        num_inactive_paths = len(self.all_paths) - len(self.active_paths)
        return num_inactive_paths * self.probe_rate

    def update(self, t, dt=0.1, current_loss=0.0):
        # Naive approach has no complex update logic for this simplified class,
        # but we step the estimator.
        # m=0.2 (restless), r=0.5 (standard TCP reset)
        self.fairness_estimator.step(m=0.2, r=0.5, loss_probability=current_loss)
        
    def get_fairness_metric(self):
        return self.fairness_estimator.get_variance()


class DynamicNaiveScheduler:
    """
    Periodically and abruptly changes active path set.
    High Responsiveness (m), Hard Resets (r).
    Decisions only at discrete epochs starting from decision_start_time.
    """
    def __init__(self, source_id, all_paths, probe_rate=0.1, active_set_size=5,
                 decision_interval=10.0, decision_start_time=100.0):
        self.source_id = source_id
        self.all_paths = [p for p in all_paths if p.source_id == source_id]
        self.probe_rate = probe_rate
        self.active_set_size = active_set_size
        
        # Deterministic sorting: RTT primary, path ID secondary for reproducibility
        self.all_paths.sort(key=lambda p: (p.rtt, p.id))
        self.active_paths = self.all_paths[:self.active_set_size]
        
        # Decision timing control
        self.decision_interval = decision_interval
        self.decision_start_time = decision_start_time
        self.last_decision_epoch = -1
        self.decision_times = []           # ← used in plotting for vertical lines
        
        # Fairness estimator (Axiom 4)
        self.fairness_estimator = FairnessEstimator()

    def get_active_paths(self):
        return self.active_paths

    def get_probing_load(self):
        num_inactive_paths = len(self.all_paths) - len(self.active_paths)
        return num_inactive_paths * self.probe_rate

    def update(self, t, dt=0.1, current_loss=0.0):
        # Always step the fairness estimator
        self.fairness_estimator.step(m=0.5, r=0.5, loss_probability=current_loss)

        # No decisions before decision_start_time
        if t < self.decision_start_time:
            return

        # Integer epoch number
        epoch = int(t // self.decision_interval)
        if epoch <= self.last_decision_epoch:
            return

        # New decision epoch
        self.last_decision_epoch = epoch
        self.decision_times.append(t)   # Record for plotting vertical lines
        print(f"Decision epoch at t={t:.1f}")

        # Perceived RTTs with reproducible noise
        perceived = [
            (p, p.rtt + np.random.uniform(-0.5, 0.5))
            for p in self.all_paths
        ]
        perceived.sort(key=lambda x: x[1])
        new_best_paths = [item[0] for item in perceived[:self.active_set_size]]

        old_set = {p.id for p in self.active_paths}
        new_set = {p.id for p in new_best_paths}

        if old_set != new_set:
            print("  -> Switching active paths!")
            self.active_paths = new_best_paths

    def get_fairness_metric(self):
        return self.fairness_estimator.get_variance()

class SmartScheduler:
    """
    PAN-aware scheduler with Hysteresis and Gradual Shifting.
    Low Responsiveness (m), Soft Resets (r).
    """
    def __init__(self, source_id, all_paths, active_set_size=5):
        self.source_id = source_id
        self.all_paths = [p for p in all_paths if p.source_id == source_id]
        self.active_set_size = active_set_size
        
        # Hysteresis
        self.SWITCH_THRESHOLD = 1.2 
        
        # Gradual Shifting
        self.is_shifting = False
        self.shift_progress = 0.0
        self.SHIFT_DURATION = 20 
        self.path_to_add = None
        self.path_to_remove = None

        self.all_paths.sort(key=lambda p: p.rtt)
        self.active_paths = self.all_paths[:self.active_set_size]
        self.last_update_time = 0
        
        # Axiom 4: Estimator
        # Smart = Low Migration (m), Soft Reset (r=1.0)
        self.fairness_estimator = FairnessEstimator()

    def get_probing_load(self):
        # Minimal probing
        probe_rate_per_path = 0.01 
        num_candidates_to_probe = 10 
        return num_candidates_to_probe * probe_rate_per_path
    
    def get_active_paths(self):
        if self.is_shifting:
            return list(set(self.active_paths + [self.path_to_add]))
        return self.active_paths

    def update(self, t, dt=0.1, current_loss=0.0):
        # Step the Fairness Estimator
        # m=0.05: Hysteresis filters out most switches, so effective migration is low
        # r=1.0: Gradual shifting preserves flow volume, acting like a perfect soft reset
        self.fairness_estimator.step(m=0.05, r=1.0, loss_probability=current_loss)

        # --- Smart Logic ---
        if self.is_shifting:
            self.shift_progress += dt
            if self.shift_progress >= self.SHIFT_DURATION:
                self.is_shifting = False
                self.active_paths.remove(self.path_to_remove)
                self.active_paths.append(self.path_to_add)
                print(f"  -> Shift complete at t={t:.1f}")
            return

        if t - self.last_update_time < 10:
            return
        self.last_update_time = t

        best_new_candidate = min((p for p in self.all_paths if p not in self.active_paths), key=lambda p: p.rtt)
        worst_active_path = max(self.active_paths, key=lambda p: p.rtt)
        
        if worst_active_path.rtt / best_new_candidate.rtt > self.SWITCH_THRESHOLD:
            print(f"Scheduler initiating a shift at t={t:.1f}")
            self.is_shifting = True
            self.shift_progress = 0.0
            self.path_to_add = best_new_candidate
            self.path_to_remove = worst_active_path
    
    def get_flow_weights(self):
        weights = {p.id: 1.0 for p in self.active_paths}
        if self.is_shifting:
            progress_ratio = self.shift_progress / self.SHIFT_DURATION
            weights[self.path_to_remove.id] = 1.0 - progress_ratio
            weights[self.path_to_add.id] = progress_ratio
        return weights

    def get_fairness_metric(self):
        return self.fairness_estimator.get_variance()