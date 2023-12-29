import numpy as np

# Generates a random configuration with outcome monotonic and additive costs
def generate_additive_configuration(m, seed, kappa=0.5, gamma=0.2, population='normal', cost_method='uniform'):
    attr = {}
    np.random.seed(seed)
    attr['seed'] = seed
    attr['m'] = m
    num_movable_nodes = []
    
    assert kappa!=0, 'Invalid kappa value'
    attr['kappa'] = kappa
    
    attr['degree_of_sparsity'] = -1

    if population == 'normal':
        unnormalized_P = np.maximum(np.random.normal(0.5, 0.1, m),0)
    elif population == 'uniform':
        unnormalized_P = np.random.rand(m)
    
    p = unnormalized_P / sum(unnormalized_P)
    attr["p"] = p
    utility = np.array(sorted((np.random.rand(m) - gamma),reverse=True))
    attr["utility"] = utility

    C=np.zeros((m,m))
    if cost_method == 'uniform':
        C[m-1,:]=np.array(sorted((1/kappa)*np.random.rand(m-1),reverse=True)+[0])
    elif cost_method == 'exponential':
        exps=np.random.exponential(scale=1, size=m-1)
        C[m-1,:]=np.array(sorted(exps, reverse=True)+[0])
        if np.floor(kappa*m) >= 1:
            scaler = C[m-1, max(0, m-1 - int(np.floor(kappa*m)))]
            C[m-1,:] /= scaler
        elif np.floor(kappa*m) == 0:
            scaler = C[m-1, m-2]-0.0000001
            C[m-1,:] /= scaler
    elif cost_method == 'normal':
        norms = np.zeros(m-1)
        for i in range(len(norms)):
            sample = np.random.normal(0.5, 0.1)
            while sample <= 0:
                sample = np.random.normal(0.5, 0.1)
            norms[i] = sample
        C[m-1,:]=np.array(sorted(norms, reverse=True)+[0])
        if np.floor(kappa*m) >= 1:
            scaler = C[m-1, max(0, m-1 - int(np.floor(kappa*m)))]
            C[m-1,:] /= scaler
        elif np.floor(kappa*m) == 0:
            scaler = C[m-1, m-2]-0.0000001
            C[m-1,:] /= scaler
    for i in reversed(range(m-1)):
        for j in range(i):
            C[i,j]=C[i+1,j]-C[i+1,i]
    
    attr['C'] = C
    for i in range(m):
        a = np.where(C[i] <= 1)
        num_movable_nodes.append(np.size(a))
    attr["num_movable_nodes"] = num_movable_nodes
    
    pi = np.zeros(m)
    attr["pi"] = pi
    return attr

# Generates a random configuration and a policy
def generate_pi_configuration(m, seed, accepted_percentage, cost_method='uniform', kappa=0.5, gamma=0.3):
    attr = {}
    np.random.seed(seed)
    attr['seed'] = seed
    attr['m'] = m
    num_movable_nodes = []

    unnormalized_P = np.maximum(np.random.normal(0.5, 0.1, m),0)
    p = unnormalized_P / sum(unnormalized_P)
    attr["p"] = p
    utility = np.array(sorted((np.random.rand(m) - gamma),reverse=True))
    attr["utility"] = utility

    if cost_method == 'uniform':
        C = np.random.rand(m, m)  # all the distances are less than 1
    elif cost_method == 'exponential':
        C = np.random.exponential(scale=1.0, size=(m, m))
        C_max = np.max(C)
        C /= C_max  # all the distances are less than 1
    elif cost_method == 'normal':
        C = np.random.normal(scale=0.5, size=(m, m))
        C_max = np.max(C)
        C /= C_max  # all the distances are less than 1
    
    # degree_of_sparsity = m - 1 if degree_of_sparsity is None else degree_of_sparsity
    degree_of_sparsity = min(int(np.ceil((1-kappa)*m)), m-1)
    attr['degree_of_sparsity'] = degree_of_sparsity # number of unreachable feature values
    attr['kappa']=kappa
    for i in range(m):
        indices = np.random.choice(m, degree_of_sparsity, replace=False)
        C[i][indices] = 2
    np.fill_diagonal(C, 0)
    attr['C'] = C

    for i in range(m):
        a = np.where(C[i] <= 1)
        num_movable_nodes.append(np.size(a))
    attr["num_movable_nodes"] = num_movable_nodes
    
    # accepted_percentage = 1 gives the optimal threshold policy in the non-strategic setting
    # accepted_percentage < 1 gives a random outcome monotonic policy
    first_negative=0
    for i in range(m):
        if utility[i]<0:
            first_negative=i
            break
    pi = np.zeros(m)
    accepted=int(accepted_percentage*first_negative)
    pi[:accepted]=1
    pi[accepted:first_negative]=sorted(np.random.rand(first_negative-accepted),reverse=True)
    attr["pi"] = pi
    return attr

# Sets attributes for a given configuration
def generate_configuration_state(U, C, Px, seed):
    attr = {}
    # np.random.seed(seed)
    attr['seed'] = seed
    attr['m'] = Px.shape[0]
    attr['C'] = C
    attr['degree_of_sparsity'] = -1
    attr['kappa']=-1
    num_movable_nodes = []
    for i in range(attr['m']):
        a = np.where(C[i] <= 1)
        num_movable_nodes.append(np.size(a))
    attr["num_movable_nodes"] = num_movable_nodes
    unnormalized_P = Px
    p = unnormalized_P / sum(unnormalized_P)
    attr["p"] = p
    attr["utility"] = U
    pi = np.zeros(attr['m'])
    for i in range(attr['m']):
        if U[i]>=0:
            pi[i]=1
    attr["pi"] = pi
    return attr


# generate_additive_configuration(m=20, seed=4, kappa=0.25, gamma=0.3, cost_method='normal')