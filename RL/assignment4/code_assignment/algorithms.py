import numpy as np 


class LinUCB:

    def __init__(self, 
        representation,
        reg_val, noise_std, delta=0.01
    ):
        self.representation = representation
        self.reg_val = reg_val
        self.noise_std = noise_std
        self.param_bound = representation.param_bound
        self.features_bound = representation.features_bound
        self.delta = delta
        self.reset()

    def reset(self):
        n_d = self.representation.features.shape[2]
        self.theta = np.zeros(n_d)
        self.inv_A = np.eye(n_d) / self.reg_val
        self.b = np.zeros(n_d)
        self.regret_bound = 0  # For question 3
        self.t = 1

    def sample_action(self, context):
        v  = self.representation.get_features(context)
        n_a, n_d = v.shape

        alpha = n_d * np.log((1 + self.t * self.features_bound**2 / self.reg_val) / self.delta)
        alpha = self.noise_std * np.sqrt(alpha) + np.sqrt(self.reg_val) * self.param_bound

        mu_hat = v @ self.theta
        U = np.zeros(n_a)
        for i in range(n_a):
            U[i] = alpha * np.sqrt(v[i] @ self.inv_A @ v[i])

        maxa = np.argmax(mu_hat + U)

        self.regret_bound += 2 * U[maxa]  # For question 3

        self.t += 1
        return maxa

    def update(self, context, action, reward):
        v = self.representation.get_features(context, action).reshape(-1, 1)

        # Sherman Morrison formula for A = A + v @ v.T 
        self.inv_A -= (self.inv_A @ v) @ (v.T @ self.inv_A) / (1 + v.T @ self.inv_A @ v)
        self.b += reward * v.flatten()
        self.theta = self.inv_A @ self.b


class RegretBalancingElim:
    def __init__(self, 
        representations,
        reg_val, noise_std,delta=0.01
    ):
        self.reg_val = reg_val
        self.noise_std = noise_std
        self.learners = [LinUCB(r, reg_val, noise_std, delta) for r in representations]  # Re use the code of LinUCB.
        self.M = len(self.learners)
        self.delta = delta
        self.last_selected_rep = None
        self.active_reps = None # list of active (non-eliminated) representations
        self.reset()

    def reset(self):
        for learner in self.learners:
            learner.reset()
        self.n = np.zeros(self.M)
        self.U = np.zeros(self.M)
        self.active_reps = list(range(self.M))
        self.last_selected_rep = None

    def sample_action(self, context):
        # Learner selection:
        self.last_selected_rep = None
        min_regret_bound = float("inf")
        for i in self.active_reps:
            if self.learners[i].regret_bound < min_regret_bound:
                self.last_selected_rep = i
                min_regret_bound = self.learners[i].regret_bound

        return self.learners[self.last_selected_rep].sample_action(context)

    def update(self, context, action, reward):
        self.learners[self.last_selected_rep].update(context, action, reward)
        self.n[self.last_selected_rep] += 1
        self.U[self.last_selected_rep] += reward

        c = 1

        bound = -float("inf")
        for j in self.active_reps:
            if self.n[j] < 2:
                continue
            bound_j = self.U[j] / self.n[j] - c * np.sqrt(np.log(self.M * np.log(self.n[j])/self.delta)/self.n[j])
            if bound_j > bound:
                bound = bound_j

        for i in self.active_reps.copy():
            if self.n[i] < 2:
                continue
            val = self.U[i] / self.n[i] + c * np.sqrt(np.log(self.M * np.log(self.n[i])/self.delta)/self.n[i])
            val += self.learners[i].regret_bound / self.n[i]
            if val < bound:
                # print("Remove", i, "at", self.n.sum())  # Log the eliminitations details.
                self.active_reps.remove(i)
