import numpy as np

from lib import configuration as Configuration
import time
import json as js
import click
from joblib import Parallel, delayed
import networkx as nx

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
        z = pi_c - C[i]
        mx_z = np.max(z)
        epsilon=1e-9
        ind = np.where(np.abs(z - mx_z) < epsilon)[0]
        max_util=-2
        for j in ind:
            if utility[j]>max_util:
                max_util=utility[j]
                mx_val=pi_c[j] * utility[j]
                br[i]=j
        u += p[i] * mx_val
    return u,br

def marginal(matching, prior_utility, prior_pi, p, C, U, candidate):
    m = prior_pi.size
    policy = prior_pi.copy()
    new_matching = matching.copy()
    policy[candidate]=1
    u = prior_utility
    for i in range(m):
        if 1-C[i,candidate] > prior_pi[matching[i]]-C[i,matching[i]]:
            u += p[i]*(U[candidate]-prior_pi[matching[i]]*U[matching[i]])
            new_matching[i]=candidate
        elif np.abs(1 - C[i,candidate] - prior_pi[matching[i]] + C[i,matching[i]])<1e-9 and U[candidate]>U[matching[i]]:
            u += p[i]*(U[candidate]-prior_pi[matching[i]]*U[matching[i]])
            new_matching[i]=candidate

    return candidate, u - prior_utility, new_matching

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
    best_utility=-1.5
    current_utility, matching = compute_utility(attr["pi"], attr["p"], attr["C"], attr["utility"])
    policy = attr["pi"].copy()
    for lim in range(m):
        if attr["utility"][lim]>=0:
            marginal_diff = marginal(matching, current_utility, policy, attr['p'], attr['C'], attr['utility'], lim)
            if current_utility + marginal_diff[1] > best_utility:
                best_lim = lim
                best_utility = current_utility + marginal_diff[1]
            policy[lim]=1
            matching = marginal_diff[2]
            current_utility += marginal_diff[1]
    
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
def compute_thres(output, C, U, Px, seed, alpha, indexing, njobs):
    
    # Configuration
    attr = Configuration.generate_configuration_state(
        U, C, Px, seed)
    m = attr["m"]
    attr["pi"] = np.zeros(m)
    
    start = time.time()
    best_utility=-1.5
    current_utility, matching = compute_utility(attr["pi"], attr["p"], attr["C"], attr["utility"])
    policy = attr["pi"].copy()
    for lim in range(m):
        if attr["utility"][lim]>=0:
            marginal_diff = marginal(matching, current_utility, policy, attr['p'], attr['C'], attr['utility'], lim)
            if current_utility + marginal_diff[1] > best_utility:
                best_lim = lim
                best_utility = current_utility + marginal_diff[1]
            policy[lim]=1
            matching = marginal_diff[2]
            current_utility += marginal_diff[1]
    
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

def optimize_component(attr):

    best_utility=-1.5
    current_utility, matching = compute_utility(attr["pi"], attr["p"], attr["C"], attr["utility"])
    policy = attr["pi"].copy()
    for lim in range(attr['m']):
        if attr["utility"][lim]>=0:
            marginal_diff = marginal(matching, current_utility, policy, attr['p'], attr['C'], attr['utility'], lim)
            if current_utility + marginal_diff[1] > best_utility:
                best_lim = lim
                best_utility = current_utility + marginal_diff[1]
            policy[lim]=1
            matching = marginal_diff[2]
            current_utility += marginal_diff[1]

    for i in range(attr['m']):
        if attr["utility"][i]>=0 and i<=best_lim:
            attr["pi"][i]=1
    
    return attr

# Performs one execution of the iterative algorithm on a given instance
# based on real data.
def compute_thres_compon(output, C, U, Px, seed, alpha, indexing, njobs):
    
    # Configuration
    attr = Configuration.generate_configuration_state(
        U, C, Px, seed)
    m = attr["m"]
    attr["pi"] = np.zeros(m)
    
    start = time.time()
    
    # Create graph based on C
    G = nx.Graph()
    G.add_nodes_from(range(m))
    for i in range(m):
        for j in range(m):
            if C[i,j]<=1:
                G.add_edge(i,j)

    # Find connected components and create attributes
    component_attrs = []
    for component in nx.connected_components(G):
        m_component = len(component) # Set
        sorted_component = sorted(component)

        indexing_component = {}
        U_component = np.zeros(m_component)
        Px_component = np.zeros(m_component)
        C_component = np.zeros((m_component, m_component))
        for ind_i, orig_i in enumerate(sorted_component):
            indexing_component[ind_i] = orig_i
            U_component[ind_i] = U[orig_i]
            Px_component[ind_i] = Px[orig_i]
            for ind_j, orig_j in enumerate(sorted_component):
                C_component[ind_i, ind_j] = C[orig_i, orig_j]

        if sum(Px_component)!=0:
            attr_component = Configuration.generate_configuration_state(
            U_component, C_component, Px_component, seed)
            attr_component["pi"] = np.zeros(m_component)

            component_attrs.append((attr_component, indexing_component))    

    # Solve independently for each component (in parallel) and merge results
    processed_attrs = Parallel(n_jobs=njobs)(delayed(optimize_component)(attr_component) \
                for attr_component, indexing_component in component_attrs)
    
    # Merge results and save them
    for ind_component, processed_attr in enumerate(processed_attrs):
        indexing_component = component_attrs[ind_component][1]
        for i in range(processed_attr['m']):
            attr['pi'][indexing_component[i]] = processed_attr['pi'][i]

    end = time.time()
    run_time = end - start

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
