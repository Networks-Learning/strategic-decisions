import numpy as np
from lib import configuration_counterfactuals as Configuration
import time
import json as js
import click
from joblib import Parallel, delayed

def dump_data(m, k, util, seed, runtime, accepted=None, avg_cost=None, best_responses=None, sparsity=None, alpha=None):
    out = {"m": m,
           "k": k, 
           "max_cover": util,
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
            max_ben=pi[i]
            max_res=i
            for j in ex:
                ben=1-C[i,j]
                if ben>max_ben:
                    max_ben=ben
                    max_res=j
            if max_res!=i:
                u+=p[i]*utility[max_res]
                br[i]=max_res
            else:
                u+=pi[i]*p[i]*utility[i]
                br[i]=i
    return u,br

# Computes the marginal difference in utility caused by adding new_ex in the set of explanations
def marginal(matching, pi, p, C, new_ex):
    
    m = pi.size
    marginal_cover = 0
    if matching is None:
        new_matching = np.full(m, fill_value=-1, dtype=int)
    else:
        new_matching = matching.copy()
    
    for i in range(m):
        if pi[i]!=1 and C[i, new_ex]<=1 and new_matching[i]==-1:
            new_matching[i] = new_ex
            marginal_cover += p[i]
            
    return new_ex, marginal_cover, new_matching

# Finds the optimal solution
def optimize(attr, k, m, njobs):

    start = time.time()

    solution_set=[]
    matching = None # Contains the best-response of each feature value at the end each iteration
    for iteration in range(k):
        print("Iteration "+str(iteration))
        marginals = Parallel(n_jobs=njobs)(delayed(marginal)(matching, attr['pi'], attr['p'], attr['C'], i)
                            for i in range(m) if attr["utility"][i]>=0 and attr["pi"][i]==1 and i not in solution_set)

        if marginals!=[]:
            max_new_exp, max_cover, max_exps = max(marginals, key=lambda x : x[1])
            solution_set.append(max_new_exp)
            matching = max_exps
    
    end = time.time()
    run_time = end - start

    final_util,final_br = compute_utility(attr["pi"], attr["p"], attr["C"], attr["utility"], solution_set)

    return final_util, final_br, run_time

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
    Runs one experiment on synthetic data using the diverse (maximum coverage) algorithm.

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
    final_util, final_br, run_time = optimize(attr, k, m, njobs)
    
    # Compute average cost incurred by changing features
    avg_cost=0
    for i in range(m):
        if final_br[i]!=i:
            avg_cost+=attr["p"][i]*attr["C"][i,final_br[i]]

    print("Diverse RunTime = " + str(run_time))
    print("Final Utility = " + str(final_util))

    # Store execution details and results
    with open(output + '_config.json', "w") as fi:
        fi.write(js.dumps(dump_data(m=m, k=k, util=final_util, seed=seed, runtime=run_time,
                                    sparsity=sparsity, accepted=accepted, avg_cost=avg_cost)))

def compute_maxcov(output, C, U, Px, k, seed, alpha, indexing, njobs=1):
    """
    Runs one experiment on real data using the diverse (maximum coverage) algorithm.

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
    """

    # Generate a configuration based on input data
    attr = Configuration.generate_configuration_state(U, C, Px, seed)
    m=attr["m"]
    
    # Find optimal solution
    final_util, final_br, run_time = optimize(attr, k, m, njobs)

    # Fix best responses to fit the real indices
    best_responses = {}
    for index, real_index in enumerate(indexing):
        best_responses[int(real_index)] = int(indexing[final_br[index]])
    
    print("Diverse RunTime = " + str(run_time))
    print("Final Utility = " + str(final_util))

    # Store execution details and results
    with open(output + '_config.json', "w") as fi:
        fi.write(js.dumps(dump_data(m=m, k=k, util=final_util, seed=seed, runtime=run_time,
                                    alpha=alpha, best_responses=best_responses)))

    return attr
    
if __name__ == '__main__':
    experiment()
