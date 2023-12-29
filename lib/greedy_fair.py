import numpy as np
from lib import configuration_counterfactuals as Configuration
import time
import json as js
import click
from joblib import Parallel, delayed

def dump_data(m, k, util, non_str_util, seed, runtime, accepted=None, best_responses=None, avg_cost=None,
                sparsity=None, alpha=None):
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
           "best_responses" : best_responses
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
def optimize(attr, k, m, njobs, num_of_partitions, partitions):

    start = time.time()
    
    partition_k = int(k/num_of_partitions)
    representation = np.zeros(num_of_partitions, dtype=int) # Contains the number of existing explanations in each partition
    
    solution_set=[]
    matching = None # Contains the best-response of each feature value at the end each iteration
    for iteration in range(k):
        print("Iteration "+str(iteration))
        marginals = Parallel(n_jobs=njobs)(delayed(marginal)(matching, attr['pi'], attr['p'], attr['C'], attr['utility'], i)
                            for i in range(m) if attr["utility"][i]>=0 and attr["pi"][i]==1 and i not in solution_set)
        # Keep only elements that do not violate the matroid constraint
        marginals = [(ex, util, match) for ex, util, match in marginals if representation[partitions[ex]]<partition_k]
        if marginals!=[]:
            max_new_exp, max_marginal, max_exps = max(marginals, key=lambda x : x[1])
            solution_set.append(max_new_exp)
            matching = max_exps
            representation[partitions[max_new_exp]]+=1 # Update the partition's representation

    end = time.time()
    run_time = end - start

    final_util,final_br = compute_utility(attr["pi"], attr["p"], attr["C"], attr["utility"], solution_set)

    return final_util, final_br, run_time
    
def compute_gf(num_of_partitions, partitions, output, C, U, Px, k, seed, alpha, indexing, njobs=1):

    # Generate a configuration based on input data
    attr = Configuration.generate_configuration_state(U, C, Px, seed)
    m=attr["m"]

    # Find optimal solution
    final_util, final_br, run_time = optimize(attr, k, m, njobs, num_of_partitions, partitions)

    # Fix best responses to fit the real indices
    best_responses = {}
    for index, real_index in enumerate(indexing):
        best_responses[int(real_index)] = int(indexing[final_br[index]])

    print("Greedy Fair RunTime = " + str(run_time))
    print("Final Utility = " + str(final_util))

    # Compute the utility gained by the optimal threshold policy in the non-strategic setting
    non_str_util=0
    for i in range(m):
        if attr["utility"][i]>0:
            non_str_util+=attr["utility"][i]*attr["p"][i]*attr["pi"][i]

    # Store execution details and results
    with open(output + '_config.json', "w") as fi:
        fi.write(js.dumps(dump_data(m=m, k=k, util=final_util, non_str_util=non_str_util, seed=seed, runtime=run_time,
                                    alpha=alpha, best_responses=best_responses)))

    return attr
    
if __name__ == '__main__':
    compute_gf()

