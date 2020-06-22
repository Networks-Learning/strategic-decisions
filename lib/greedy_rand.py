import numpy as np
from lib import configuration as Configuration
import time
import json as js
import click
from joblib import Parallel, delayed
import random

def dump_data(m, k, util, seed, runtime, pi=None, best_responses=None, avg_cost=None,
                sparsity=None, alpha=None, leaking_results=None):
    out = {"m": m,
           "k": k, 
           "greedy_rand": util,
           "seed": seed,
           "time": runtime,
           "sparsity": sparsity,
           "avg_cost": avg_cost,
           "alpha": alpha,
           "best_responses" : best_responses,
           "pi" : pi,
           "leaking_results" : leaking_results
           }
    out = {k:v for k,v in out.items() if v is not None}
    return out

# Computes the utility given a set of explanations
def compute_utility(p, C, utility, ex):
    m = p.size
    u = 0
    br = np.arange(m) # best-responses
    pi = np.zeros(m, dtype=int)
    ex_filt=[]
    # Remove dummy explanations
    for j in ex:
        if j<m:
            pi[j]=1
            ex_filt.append(j)
    for i in range(m):
        if pi[i]!=1:
            max_util = -np.inf
            for j in ex_filt:
                if C[i,j]<=1 and utility[j]>max_util and utility[j]>utility[i]:
                    max_util=utility[j]
                    max_j=j
            if max_util!=-np.inf:
                u+=p[i]*utility[max_j]
                br[i]=max_j
            elif utility[i]>=0:
                u+=p[i]*utility[i]
                br[i]=i
                pi[i]=1
        else:
            br[i]=i
            u+=p[i]*utility[i]
    return u,br,pi

# Computes the marginal difference in utility caused by adding new_ex in the set of explanations
def marginal(matching, pi, p, C, U, explanations, new_ex):
    
    m = p.size
    marginal_util = 0
    if explanations==[]:
        new_matching = np.arange(m)
        new_pi = np.zeros(m, dtype=int)
        new_pi[U>=0]=1
    else:
        new_matching = matching.copy()
        new_pi = pi.copy()
    
    for i in range(m):
        if i not in explanations and i!=new_ex and 1-C[i,new_ex]>=0 and U[new_ex]>=U[new_matching[i]]:
            marginal_util += p[i]*(U[new_ex]-new_pi[new_matching[i]]*U[new_matching[i]])
            new_matching[i] = new_ex
            new_pi[i] = 0
        elif i==new_ex and U[i]>=0:
            marginal_util += p[i]*(U[new_ex]-new_pi[new_matching[i]]*U[new_matching[i]])
            new_matching[i] = i
            new_pi[i] = 1
    
    return new_ex, marginal_util, new_matching, new_pi

# Finds the optimal solution
def optimize(attr, k, m, njobs, seed, leaking):

    random.seed(seed)
    start = time.time()

    solution_set=[]
    matching = None # Contains the best-response of each feature value at the end each iteration
    pi = None
    for iteration in range(k):
        print("Iteration "+str(iteration))
        marginals = Parallel(n_jobs=njobs)(delayed(marginal)(matching, pi, attr['p'], attr['C'], attr['utility'],
                            solution_set, i) for i in range(m) if attr["utility"][i]>=0 and i not in solution_set)
        
        for i in range(m,m+2*k):  # DUMMIES
            if i not in solution_set:
                marginals.append((i, 0, matching, pi))

        topk = np.argpartition(np.array([x[1] for x in marginals]), -k)[-k:]
        chosen_one = random.choice(topk)
        solution_set.append(marginals[chosen_one][0])
        matching = marginals[chosen_one][2]
        pi = marginals[chosen_one][3]

    solution_set_filter=[]
    for j in solution_set:
        if j<m:
            solution_set_filter.append(j)

    end = time.time()
    run_time = end - start

    final_util,final_br,attr['pi'] = compute_utility(attr["p"], attr["C"], attr["utility"], solution_set_filter)

    # If needed, each feature value learns one extra counterfactual explanation from the solution set and chooses to best-responde
    if leaking is not None:
        leaking_results={}
        for pr in leaking:
            leak_util=0
            leak_br=np.arange(m)
            for i in range(m):
                if attr["pi"][i]==1:
                    leak_br[i]=i
                else:
                    if random.random()<pr:
                        leaked=random.choice(solution_set_filter)
                        if attr["C"][i,leaked]<=1 and 1-attr["C"][i,leaked]>=attr["pi"][final_br[i]]-attr["C"][i,final_br[i]]:
                            leak_br[i]=leaked
                        else:
                            leak_br[i]=final_br[i]
                    else:
                        leak_br[i]=final_br[i]    
                leak_util+=attr["utility"][leak_br[i]]*attr["pi"][leak_br[i]]*attr["p"][i]
            leaking_results[pr]=leak_util
    else:
        leaking_results=None

    return final_util, final_br, attr['pi'], run_time, leaking_results

@click.command()
@click.option('--output', required=True, help="output directory")
@click.option('--m', default=4, help="Number of states")
@click.option('--k', default=2, help="Number of examples")
@click.option('--seed', default=2, help="random number for seed.")
@click.option('--sparsity', default=2, help="sparsity of the graph")
@click.option('--njobs', default=1, help="number of parallel threads")
def experiment(output, m, k, seed, sparsity, njobs):
    """
    Runs one experiment on synthetic data using the greedy randomized algorithm.

    Parameters
    ----------
    output : string
        output directory prefix (e.g., outputs/exec1_)
    m : int
        number of feature values
    k : int
        maximum number of explanations
    seed : int
        random seed for reproducibility
    sparsity : int
        number of unreachable feature values 
    njobs : int
        number of parallel threads to be used
    """

    # Generate a random configuration
    attr = Configuration.generate_pi_configuration(
        m, seed, accepted_percentage=1, degree_of_sparsity=sparsity)
    
    # Find optimal solution
    final_util, final_br, final_pi, run_time, _ = optimize(attr, k, m, njobs, seed, [])
    
    # Compute average cost incurred by changing features
    avg_cost=0
    for i in range(m):
        if final_br[i]!=i:
            avg_cost+=attr["p"][i]*attr["C"][i,final_br[i]]

    print("Greedy Randomized RunTime = " + str(run_time))
    print("Final Utility = " + str(final_util))

    # Store execution details and results
    with open(output + '_config.json', "w") as fi:
        fi.write(js.dumps(dump_data(m=m, k=k, util=final_util, seed=seed, runtime=run_time,
                                    sparsity=sparsity, avg_cost=avg_cost)))


def compute_gr(output, C, U, Px, k, seed, alpha, indexing, njobs=1, leaking=None):
    """
    Runs one experiment on real data using the greedy randomized algorithm.

    Parameters
    ----------
    output : string
        output directory prefix (e.g., outputs/exec1_)
    C : numpy array
        pairwise costs
    U : numpy array
        utility gained by each feature value
    Px : numpy array
        initial population in each feature value
    k : int
        maximum number of explanations
    seed : int 
        random seed for reproducibility
    alpha : float
        alpha parameter value
    indexing : numpy array
        real indices of feature values (unsorted)
    njobs : int
        number of parallel threads to be used
    leaking : numpy array
        leakage probabilities
    """

    # Generate a configuration based on input data
    attr = Configuration.generate_configuration_state(U, C, Px, seed)
    m=attr["m"]

    # Find optimal solution
    final_util, final_br, final_pi, run_time, leaking_results = optimize(attr, k, m, njobs, seed, leaking)
    
    # Fix best responses to fit the real indices
    best_responses = {}
    for index, real_index in enumerate(indexing):
        best_responses[int(real_index)] = int(indexing[final_br[index]])

    # Fix the policy to fit the real indices
    pi = {}
    for index, real_index in enumerate(indexing):
        pi[int(real_index)] = int(final_pi[index])

    print("Greedy Randomized RunTime = " + str(run_time))
    print("Final Utility = " + str(final_util))

    # Store execution details and results
    with open(output + '_config.json', "w") as fi:
        fi.write(js.dumps(dump_data(m=m, k=k, util=final_util, seed=seed, runtime=run_time,
                                    alpha=alpha, best_responses=best_responses, pi=pi, leaking_results=leaking_results)))
    
    return attr

if __name__ == '__main__':
    experiment()
