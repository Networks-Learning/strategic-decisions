import numpy as np

from lib import configuration as Configuration
import time
import json as js
import click
from joblib import Parallel, delayed

def dump_data(attr, strategic_threshold, time, pi, strategic_threshold_br, alpha=None):
    out = {
        # Configuration
        "m": attr["m"],
        "seed": attr['seed'],
        "sparsity": attr['degree_of_sparsity'],
        "kappa": attr['kappa'],
        "alpha": alpha,
        # Execution details
        "strategic_threshold": strategic_threshold,
        "time": time,
        "pi" : pi,
        "strategic_threshold_br" : strategic_threshold_br,
    }
    out = {k:v for k,v in out.items() if v is not None}
    return out

# Computes the utility of a given policy and the best-response
# of the individuals of each feature value.
def compute_utility(pi_c, p, C, utility):
    m = pi_c.size
    u = 0
    br = np.zeros(m, dtype=int)
    for i in range(m):
        # TODO: Make this part quicker
        z = pi_c - C[i]
        mx_z = np.max(z)
        epsilon=1e-9
        ind = np.where(np.abs(z - mx_z) < epsilon)[0]
        max_util=-2
        #
        for j in ind:
            if utility[j]>max_util:
                max_util=utility[j]
                mx_val=pi_c[j] * utility[j]
                br[i]=j
        u += p[i] * mx_val
    return u,br


# Performs one execution of the shifted threshold algorithm on a randomly generated
# instance given the following parameters as command line arguments.
@click.command()
@click.option('--output', required=True, help="output directory")
@click.option('--m', default=4, type=int, help="Number of states")
@click.option('--seed', default=2, type=int, help="random number for seed.")
@click.option('--sparsity', default=2, type=int, help="sparsity of the graph")
@click.option('--kappa', default=0.2, type=float, help="inverse sparsity of the graph")
@click.option('--gamma', default=0.2, type=float, help="gamma parameter")
@click.option('--additive', is_flag=True, default=False, help="if used, it generates additive configuration")
def experiment(output, m, seed, sparsity, gamma, kappa, additive):
    if additive:
        attr = Configuration.generate_additive_configuration(
            m, seed, kappa=kappa, gamma=gamma)
    else:
        attr = Configuration.generate_pi_configuration(
            m, seed, accepted_percentage=1, degree_of_sparsity=sparsity, gamma=gamma)
        attr["pi"] = np.zeros(m)
    
    start = time.time()
    best_utility = -1.5 
    for lim in range(m):
        if attr["utility"][lim]>=0:
            policy = np.zeros(m)
            for i in range(m):
                if attr["utility"][i]>=0 and i<=lim:
                    policy[i]=1
            utility = compute_utility(
                    policy, attr["p"], attr["C"], attr["utility"])[0]
            if utility>best_utility:
                best_lim = lim
                best_utility = utility
    
    end = time.time()
    run_time = end - start
    
    for i in range(m):
        if attr["utility"][i]>=0 and i<=best_lim:
            attr["pi"][i]=1

    u,br = compute_utility(attr["pi"], attr["p"], attr["C"], attr["utility"])
    
    print("Threshold RunTime = " + str(run_time))
    print("Final Utility = " + str(u))

    br = {ind:int(x) for ind,x in enumerate(br)}
    pi = {ind:float(x) for ind,x in enumerate(attr["pi"])}

    with open(output + '_config.json', "w") as fi:
        fi.write(js.dumps(dump_data(attr=attr, strategic_threshold=u, time=run_time,
                                    pi=pi, strategic_threshold_br=br)))

# Performs one execution of the iterative algorithm on a given instance
# based on real data.
def compute_thres(output, C, U, Px, seed, alpha, indexing):
    
    # Configuration
    attr = Configuration.generate_configuration_state(
        U, C, Px, seed)
    m = attr["m"]
    attr["pi"] = np.zeros(m)
    
    start = time.time()
    best_utility = -1.5 
    for lim in range(m):
        if attr["utility"][lim]>=0:
            policy = np.zeros(m)
            for i in range(m):
                if attr["utility"][i]>=0 and i<=lim:
                    policy[i]=1
            utility = compute_utility(
                    policy, attr["p"], attr["C"], attr["utility"])[0]
            if utility>best_utility:
                best_lim = lim
                best_utility = utility
    
    end = time.time()
    run_time = end - start
    
    for i in range(m):
        if attr["utility"][i]>=0 and i<=best_lim:
            attr["pi"][i]=1

    u,br = compute_utility(attr["pi"], attr["p"], attr["C"], attr["utility"])
    
    print("Threshold RunTime = " + str(run_time))
    print("Final Utility = " + str(u))

    # Fix best responses to fit the real indices
    best_responses = {}
    for index, real_index in enumerate(indexing):
        best_responses[int(real_index)] = int(indexing[br[index]])

    # Fix the policy to fit the real indices
    pi = {}
    for index, real_index in enumerate(indexing):
        pi[int(real_index)] = attr["pi"][index]
    
    with open(output + '_config.json', "w") as fi:
        fi.write(js.dumps(dump_data(attr=attr, strategic_threshold=u, time=run_time,
                                    pi=pi, strategic_threshold_br=best_responses, alpha=alpha)))

    return attr

if __name__ == '__main__':
    experiment()