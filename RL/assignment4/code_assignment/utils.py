import numpy as np
from sklearn.decomposition import PCA


class ContRepresentation:

    def predict(self, context, action):
        pass

    def sample_context(self):
        pass

    def optimal_reward(self, context):
        pass


class FiniteContLinearRep(ContRepresentation):

    def __init__(self, features, param, contexts=None, actions=None):
        assert len(features.shape) == 3
        nc, na, nd = features.shape
        assert len(param) == nd
        if contexts is not None:
            assert contexts.shape[0] == nc
        else:
            contexts = np.arange(nc)
        if actions is not None:
            assert actions.shape[0] == na
        else:
            actions = np.arange(na)
        self.features = features
        self.param = param
        self.contexts = contexts
        self.actions = actions
        self.features_bound = np.linalg.norm(features, 2, axis=2).max()
        self.param_bound = np.linalg.norm(param, 2)
    
    def predict(self, context, action):
        y = self.features[context, action] @ self.param
        return y

    def get_features(self, context, action=None):
        if action is None:
            return self.features[context]
        else:
            return self.features[context, action]

    def sample_context(self):
        nc = self.contexts.shape[0]
        cont = np.random.choice(nc)
        return self.contexts[cont]

    def optimal_reward(self, context):
        y = self.features[context] @ self.param
        return y.max()

    def num_actions(self):
        return self.features.shape[1]

    def dim(self):
        return self.features.shape[2]


class ContBanditProblem:

    def __init__(self, reward_rep, algo, noise_std):
        self.reward_rep=reward_rep
        self.algo = algo
        self.noise_std = noise_std
        self.t = 0
        self.reset()
    
    def sample_reward(self, context, action):
        noise = np.random.randn(1) * self.noise_std
        rew = self.reward_rep.predict(context, action) + noise
        return rew
    
    def reset(self):
        self.t = 0
        self.algo.reset()

    def run(self, horizon):
        self.exp_instant_regret = np.zeros(horizon)
        self.instant_regret = np.zeros(horizon)
        while self.t<horizon:
            context = self.reward_rep.sample_context()
            action = self.algo.sample_action(context)
            reward = self.sample_reward(context, action)
            self.algo.update(context, action, reward)

            pseudo_reg, optimal_rew = self.pseudo_regret(context, action)
            self.exp_instant_regret[self.t] = pseudo_reg
            self.instant_regret[self.t] = optimal_rew - reward

            self.t += 1

    def pseudo_regret(self, context, action):
        exp_rew = self.reward_rep.predict(context, action)
        opt_rew = self.reward_rep.optimal_reward(context)
        return opt_rew - exp_rew, opt_rew



def make_random_rep(n_contexts, n_arms, dim, ortho=True, normalize=True):
    features = np.random.normal(size=(n_contexts, n_arms, dim))
    param = 2 * np.random.uniform(size=dim) - 1
    
    #Orthogonalize features
    if ortho:
        features = np.reshape(features, (n_contexts * n_arms, dim))
        orthogonalizer = PCA(n_components=dim) #no dimensionality reduction
        features = orthogonalizer.fit_transform(features)
        features = np.reshape(features, (n_contexts, n_arms, dim))
        features = np.take(features, np.random.permutation(dim), axis=2)
    
    r1 = FiniteContLinearRep(features, param)
    
    if normalize:
        r1 = normalize_param(r1)
        
    return r1

#Transforming representations
def normalize_param(rep, scale=1.):
    param = rep.param
    param_norm = np.linalg.norm(param)
    param = param / param_norm * scale
    
    features = rep.features * param_norm / scale
    
    return FiniteContLinearRep(features, param)


def make_newlinrep_reshaped(linrep, new_dim):
    nc, na, nd = linrep.features.shape
    if new_dim > nd:
        rand_feat = np.random.randn(nc,na, new_dim-nd)
        new_feat = np.concatenate((linrep.features,rand_feat), axis=-1)
        new_param = np.concatenate((linrep.param, np.zeros(new_dim-nd)))
    elif new_dim < nd:
        new_feat = linrep.features[:,:, :new_dim]
        new_param = linrep.param[:new_dim]
    else:
        new_feat = linrep.features
        new_param = linrep.param
    assert (nc, na, new_dim) == new_feat.shape
    assert new_dim == new_param.shape[0]
    return FiniteContLinearRep(new_feat, new_param)

