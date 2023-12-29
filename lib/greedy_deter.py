import numpy as np
from lib import configuration_counterfactuals as Configuration
import time
import json as js
import click
from joblib import Parallel, delayed

def dump_data(m, k, util, non_str_util, seed, runtime, accepted=None, best_responses=None, avg_cost=None,
                sparsity=None, alpha=None, leaking_results=None):
    out = {"m": m,
           "k": k, 
           "strategic": util,
           "non_strategic": non_str_util,
           "seed": seed,
           "time": runtime,
           "sparsity": sparsity,
           "accepted": accepted,
           "avg_cost": avg_cost,
           "alpha": alpha,
           "best_responses" : best_responses,
           "leaking_results" : leaking_results
           }
    out = {k:v for k,v in out.items() if v is not None}
    return out

# Computes the utility given a policy and a set of explanations
def compute_utility(pi, p, C, utility, ex):
    m = pi.size
    u = 0
    br = np.arange(m) # best-responses
    for i in range(m):
        if pi[i]==1:
            u+=p[i]*utility[i]
            br[i]=i
        else:
            max_util=-np.inf
            for j in ex:
                if 1-C[i,j]>=pi[i] and utility[j]>max_util:
                    max_util=utility[j]
                    max_j=j
            if max_util!=-np.inf:
                u+=p[i]*utility[max_j]
                br[i]=max_j
            else:
                u+=pi[i]*p[i]*utility[i]
                br[i]=i
    return u,br

# Computes the marginal difference in utility caused by adding new_ex in the set of explanations
def marginal(matching, pi, p, C, U, new_ex):
    
    m = pi.size
    marginal_util = 0
    if matching is None:
        new_matching = np.arange(m)
    else:
        new_matching = matching.copy()
    
    for i in range(m):
        if pi[i]!=1 and 1-C[i,new_ex]>=pi[i] and U[new_ex]>=U[new_matching[i]]:
            marginal_util += p[i]*(U[new_ex]-pi[new_matching[i]]*U[new_matching[i]])
            new_matching[i] = new_ex
    
    return new_ex, marginal_util, new_matching

# Finds the optimal solution
def optimize(attr, k, m, njobs, leaking):

    start = time.time()

    solution_set=[]
    matching = None # Contains the best-response of each feature value at the end each iteration
    for iteration in range(k):
        print("Iteration "+str(iteration))
        marginals = Parallel(n_jobs=njobs)(delayed(marginal)(matching, attr['pi'], attr['p'], attr['C'], attr['utility'], i)
                            for i in range(m) if attr["utility"][i]>=0 and attr["pi"][i]==1 and i not in solution_set)
        
        if marginals!=[]:
            max_new_exp, max_marginal, max_exps = max(marginals, key=lambda x : x[1])
            solution_set.append(max_new_exp)
            matching = max_exps

    end = time.time()
    run_time = end - start

    final_util,final_br = compute_utility(attr["pi"], attr["p"], attr["C"], attr["utility"], solution_set)

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
                    if np.random.rand()<pr:
                        leaked=np.random.choice(solution_set)
                        if attr["C"][i,leaked]<=1 and 1-attr["C"][i,leaked]>=attr["pi"][final_br[i]]-attr["C"][i,final_br[i]]:
                            leak_br[i]=leaked
                        else:
                            leak_br[i]=final_br[i]
                    else:
                        leak_br[i]=final_br[i]    
                leak_util+=attr["utility"][leak_br[i]]*attr["pi"][leak_br[i]]*attr["p"][i]
            leaking_results[pr]=leak_util
    else:
        leaking_results = None
    
    return final_util, final_br, run_time, leaking_results

@click.command()
@click.option('--output', required=True, help="output directory")
@click.option('--m', default=4, help="Number of states")
@click.option('--k', default=2, help="Number of examples")
@click.option('--seed', default=2, help="random number for seed.")
@click.option('--sparsity', default=2, help="sparsity of the graph")
@click.option('--accepted', default=1, type=float, help="percentage with pi equals one")
@click.option('--njobs', default=1, help="number of parallel threads")
def experiment(output, m, k, seed, sparsity, accepted, njobs):
    """
    Runs one experiment on synthetic data using the greedy deterministic algorithm.

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
    accepted : float
        percentage of feature values with non-negative utility to set pi to 1
    njobs : int
        number of parallel threads to be used
    """

    # Generate a random configuration
    attr = Configuration.generate_pi_configuration(
        m, seed, accepted_percentage=accepted, degree_of_sparsity=sparsity)
    
    # Find optimal solution
    final_util, final_br, run_time, _ = optimize(attr, k, m, njobs, [])

    # Compute average cost incurred by changing features
    avg_cost=0
    for i in range(m):
        if final_br[i]!=i:
            avg_cost+=attr["p"][i]*attr["C"][i,final_br[i]]

    print("Greedy Deterministic RunTime = " + str(run_time))
    print("Final Utility = " + str(final_util))

    # Compute the utility gained by the optimal threshold policy in the non-strategic setting
    non_str_util=0
    for i in range(m):
        if attr["utility"][i]>0:
            non_str_util+=attr["utility"][i]*attr["p"][i]*attr["pi"][i]

    # Store execution details and results
    with open(output + '_config.json', "w") as fi:
        fi.write(js.dumps(dump_data(m=m, k=k, util=final_util, non_str_util=non_str_util, seed=seed, runtime=run_time,
                                    sparsity=sparsity, accepted=accepted, avg_cost=avg_cost)))
    
def compute_gd(output, C, U, Px, k, seed, alpha, indexing, leaking=None, njobs=1):
    """
    Runs one experiment on real data using the greedy deterministic algorithm.

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
    leaking : numpy array
        leakage probabilities
    njobs : int
        number of parallel threads to be used
    """

    # Generate a configuration based on input data
    attr = Configuration.generate_configuration_state(U, C, Px, seed)
    m=attr["m"]

    # Find optimal solution
    final_util, final_br, run_time, leaking_results = optimize(attr, k, m, njobs, leaking)

    # Fix best responses to fit the real indices
    best_responses = {}
    for index, real_index in enumerate(indexing):
        best_responses[int(real_index)] = int(indexing[final_br[index]])

    print("Greedy Deterministic RunTime = " + str(run_time))
    print("Final Utility = " + str(final_util))

    # Compute the utility gained by the optimal threshold policy in the non-strategic setting
    non_str_util=0
    for i in range(m):
        if attr["utility"][i]>0:
            non_str_util+=attr["utility"][i]*attr["p"][i]*attr["pi"][i]

    # Store execution details and results
    with open(output + '_config.json', "w") as fi:
        fi.write(js.dumps(dump_data(m=m, k=k, util=final_util, non_str_util=non_str_util, seed=seed, runtime=run_time,
                                    alpha=alpha, best_responses=best_responses, leaking_results=leaking_results)))

    return attr
if __name__ == '__main__':
    experiment()

