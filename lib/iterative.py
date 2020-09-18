import numpy as np

from lib import configuration as Configuration
import time
import json as js
import click
from joblib import Parallel, delayed
import networkx as nx

def dump_data(attr, non_strategic, strategic, strategic_deter, time, iterations, pi, pi_non_strategic,
            pi_strategic_deter, non_strategic_br, strategic_br, strategic_deter_br, components=None, alpha=None):
    out = {
        # Configuration
        "m": attr["m"],
        "seed": attr['seed'],
        "sparsity": attr['degree_of_sparsity'],
        "kappa": attr['kappa'],
        "parallel": attr['parallel'],
        "split_components" : attr['split_components'],
        "alpha": alpha,
        # Execution details
        "strategic": strategic,
        "non_strategic": non_strategic,
        "strategic_deter": strategic_deter,
        "iterations": iterations,
        "components": components,
        "time": time,
        "pi" : pi,
        "pi_non_strategic": pi_non_strategic,
        "pi_strategic_deter": pi_strategic_deter,
        "non_strategic_br" : non_strategic_br,
        "strategic_br": strategic_br,
        "strategic_deter_br": strategic_deter_br
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

# Updates the policy value of the kth feature value considering
# all the critical values and choosing the one that maximizes utility.
def update(k, pi, p, C, utility):

    m = pi.size
    pi_c = pi.copy()
    pi_c[k] = 0
    previous_u, previous_br = compute_utility(pi_c, p, C, utility)
    if utility[k]<0:
        return [0, -np.inf, previous_br]
    
    candidate_values = {}
    for i in range(m):
        threshold = np.around(pi_c[previous_br[i]] - C[i,previous_br[i]] + C[i,k], 9)
        if threshold not in candidate_values:
            candidate_values[threshold]=[]
        candidate_values[threshold].append(i)
    candidate_values = {k:v for k,v in candidate_values.items() if (k > 0 and k < 1)}
    
    # Special case 0
    previous_v = 0
    previous_pop = sum([p[ind] for ind, x in enumerate(previous_br) if previous_br[ind]==k])
    best_possible_utility = previous_u
    best_value = 0
    best_br = previous_br

    future_shifters=[i for i in range(m) if np.abs(pi_c[previous_br[i]] - C[i,previous_br[i]] + C[i,k])<1e-9 and 
                        previous_br[i]!=k and utility[k] <= utility[previous_br[i]]]
    
    # Intermediate values
    for v in sorted(list(candidate_values.keys())):
        
        previous_u += previous_pop*(v-previous_v)*utility[k]

        for s in future_shifters:
            previous_u += p[s]*(v*utility[k] - pi_c[previous_br[s]]*utility[previous_br[s]])
            previous_br[s] = k
            previous_pop += p[s]

        future_shifters=[]
        shifters = candidate_values[v]
        u = previous_u
        br = previous_br.copy()
        for s in shifters:
            if utility[k] > utility[previous_br[s]]:
                u += p[s]*(v*utility[k] - pi_c[previous_br[s]]*utility[previous_br[s]])
                br[s] = k
                previous_pop += p[s]
            else:
                future_shifters.append(s)
        
        if u > best_possible_utility:
            best_possible_utility = u
            best_value = v
            best_br=br
        
        previous_u=u
        previous_v=v
        previous_br=br.copy()

    # Special case 1
    pi_c[k] = 1
    u,br = compute_utility(pi_c, p, C, utility)
    if u > best_possible_utility:
        best_possible_utility = u
        best_value = 1
        best_br=br
    
    
    return [best_value, best_possible_utility, best_br]


# Performs one execution of the iterative algorithm on a randomly generated
# instance given the following parameters as command line arguments.
@click.command()
@click.option('--njobs', required=True, type=int, help="number of parallel threads")
@click.option('--output', required=True, help="output directory")
@click.option('--m', default=4, type=int, help="Number of states")
@click.option('--max_iter', default=20, type=int)
@click.option('--seed', default=2, type=int, help="random number for seed.")
@click.option('--sparsity', default=2, type=int, help="sparsity of the graph")
@click.option('--kappa', default=0.2, type=float, help="inverse sparsity of the graph")
@click.option('--gamma', default=0.2, type=float, help="gamma parameter")
@click.option('--additive', is_flag=True, default=False, help="if used, it generates additive configuration")
def experiment(output, m, seed, sparsity, gamma, kappa, additive, max_iter, njobs):
    if additive:
        attr = Configuration.generate_additive_configuration(
            m, seed, kappa=kappa, gamma=gamma)
    else:
        attr = Configuration.generate_pi_configuration(
            m, seed, accepted_percentage=1, degree_of_sparsity=sparsity, gamma=gamma)
        attr["pi"] = np.zeros(m)
    
    best_utility = -1.5 
    iterations = 0

    start = time.time()
    parallel = True if njobs > 1 else False
    attr['parallel'] = parallel
    attr['split_components'] = False
    while True:
        any_update = False
        if parallel:
            previous_pi = attr["pi"].copy()
            results = Parallel(n_jobs=njobs)(delayed(lambda x: update(
                x, previous_pi, attr["p"], attr["C"], attr["utility"]))(k) for k in range(m))
            for (pi_k, best_util_k, best_br_k), k in zip(results, list(range(m))):
                if best_util_k > best_utility:
                    attr["pi"][k] = pi_k
                    any_update = True
            best_utility = compute_utility(
                attr["pi"], attr["p"], attr["C"], attr["utility"])[0]
        else:
            for k in range(m):
                [best_value, best_possible_utility, best_possible_responses] = update(
                    k, attr["pi"], attr["p"], attr["C"], attr["utility"])
                if best_possible_utility > best_utility:
                    attr["pi"][k] = best_value
                    best_utility = best_possible_utility
                    best_responses = best_possible_responses
                    any_update = True
        print("Step = " + str(iterations+1))
        print("Iteration utility = " + str(best_utility))
        iterations += 1

        if not any_update or (parallel and iterations >= max_iter):
            end = time.time()
            run_time = end - start
            u,br=compute_utility(attr['pi'],attr['p'],attr['C'],attr['utility'])
            print("Iterative RunTime = " + str(run_time))
            print("Final Utility = " + str(u))
            break

    pi_non_strategic = np.zeros(attr['m'])
    non_strategic_br = np.arange(attr['m'],dtype=int)
    non_strategic_utility = 0
    for i in range(m):
        if attr['utility'][i]>=0:
            pi_non_strategic[i]=1
            non_strategic_utility += attr['p'][i]*attr['utility'][i]

    pi_strategic_deter = attr["pi"].copy()
    pi_strategic_deter[pi_strategic_deter > 0.5] = 1
    pi_strategic_deter[pi_strategic_deter <= 0.5] = 0

    strategic_deterministic_utility, strategic_deterministic_br = compute_utility(
        pi_strategic_deter, attr["p"], attr["C"], attr["utility"])

    br = {ind:int(x) for ind,x in enumerate(br)}
    non_strategic_br = {ind:int(x) for ind,x in enumerate(non_strategic_br)}
    strategic_deterministic_br = {ind:int(x) for ind,x in enumerate(strategic_deterministic_br)}
    pi = {ind:float(x) for ind,x in enumerate(attr["pi"])}
    pi_non_strategic = {ind:float(x) for ind,x in enumerate(pi_non_strategic)}
    pi_strategic_deter = {ind:float(x) for ind,x in enumerate(pi_strategic_deter)}

    with open(output + '_config.json', "w") as fi:
        fi.write(js.dumps(dump_data(attr=attr, non_strategic=non_strategic_utility, strategic=best_utility,
                                    strategic_deter=strategic_deterministic_utility, time=run_time, iterations=iterations,
                                    pi=pi, pi_non_strategic=pi_non_strategic, pi_strategic_deter=pi_strategic_deter,
                                    non_strategic_br=non_strategic_br, strategic_br=br,
                                    strategic_deter_br=strategic_deterministic_br)))


# Performs one execution of the iterative algorithm on a connected component
# of the real data
def optimize_component(attr, max_iter, parallel, njobs):

    best_utility = -1.5 # initial
    iterations = 0
    while True:
        any_update = False
        if parallel:
            previous_pi = attr["pi"].copy()
            results = Parallel(n_jobs=njobs)(delayed(lambda x: update(
                x, previous_pi, attr["p"], attr["C"], attr["utility"]))(k) for k in range(attr['m']))
            for (pi_k, best_util_k, best_br_k), k in zip(results, list(range(attr['m']))):
                if best_util_k > best_utility:
                    attr["pi"][k] = pi_k
                    any_update = True
            best_utility = compute_utility(
                attr["pi"], attr["p"], attr["C"], attr["utility"])[0]
        else:
            for k in range(attr['m']):
                [best_value, best_possible_utility, best_possible_responses] = update(
                    k, attr["pi"], attr["p"], attr["C"], attr["utility"])
                if best_possible_utility > best_utility:
                    attr["pi"][k] = best_value
                    best_utility = best_possible_utility
                    best_responses = best_possible_responses
                    any_update = True
        iterations += 1

        if not any_update or (parallel and iterations >= max_iter):
            break
    
    attr['iterations']=iterations
    return attr

# Finds the connected components of the graph and executes the iterative algorithm on each one
def compute_iter(output, C, U, Px, seed, alpha, indexing, max_iter=20, verbose=False, njobs=1, split_components=True):
    
    # Configuration
    attr = Configuration.generate_configuration_state(
        U, C, Px, seed)
    m = attr["m"]
    attr["pi"] = np.zeros(m)
    parallel = True if njobs > 1 else False
    attr['parallel'] = parallel
    attr['split_components'] = split_components
    start = time.time()
    
    if split_components:

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

        num_components = len(component_attrs)

        # Solve independently for each component (in parallel) and merge results
        # processed_attrs = Parallel(n_jobs=njobs)(delayed(optimize_component)(attr_component, max_iter) \
        #             for attr_component, indexing_component in component_attrs)

        # Solve independently for each component
        processed_attrs = []
        for attr_component, indexing_component in component_attrs:
            processed_attrs.append(optimize_component(attr_component, max_iter, parallel, njobs))

        # Merge results and save them
        for ind_component, processed_attr in enumerate(processed_attrs):
            indexing_component = component_attrs[ind_component][1]
            for i in range(processed_attr['m']):
                attr['pi'][indexing_component[i]] = processed_attr['pi'][i]
        
        iterations = np.mean([processed_attr['iterations'] for processed_attr in processed_attrs])

    else:

        attr = optimize_component(attr, max_iter, parallel, njobs)
        iterations = attr['iterations']
        num_components = 1
    
    end = time.time()
    run_time = end - start
    u,br=compute_utility(attr['pi'],attr['p'],attr['C'],attr['utility'])
    print("Iterative RunTime = " + str(run_time))
    print("Final Utility = " + str(u))
    

    pi_non_strategic = np.zeros(attr['m'])
    non_strategic_br = np.arange(attr['m'],dtype=int)
    non_strategic_utility = 0
    for i in range(m):
        if attr['utility'][i]>=0:
            pi_non_strategic[i]=1
            non_strategic_utility += attr['p'][i]*attr['utility'][i]

    pi_strategic_deter = attr["pi"].copy()
    pi_strategic_deter[pi_strategic_deter > 0.5] = 1
    pi_strategic_deter[pi_strategic_deter <= 0.5] = 0

    strategic_deterministic_utility, strategic_deterministic_br = compute_utility(
        pi_strategic_deter, attr["p"], attr["C"], attr["utility"])

    # Fix best responses to fit the real indices
    best_responses = {}
    for index, real_index in enumerate(indexing):
        best_responses[int(real_index)] = int(indexing[br[index]])
    
    best_responses_non_strategic = {}
    for index, real_index in enumerate(indexing):
        best_responses_non_strategic[int(real_index)] = int(indexing[non_strategic_br[index]])

    best_responses_strategic_deterministic = {}
    for index, real_index in enumerate(indexing):
        best_responses_strategic_deterministic[int(real_index)] = int(indexing[strategic_deterministic_br[index]])

    # Fix the policy to fit the real indices
    pi = {}
    for index, real_index in enumerate(indexing):
        pi[int(real_index)] = attr["pi"][index]
    
    non_strategic_pi = {}
    for index, real_index in enumerate(indexing):
        non_strategic_pi[int(real_index)] = pi_non_strategic[index]
    
    strategic_deter_pi = {}
    for index, real_index in enumerate(indexing):
        strategic_deter_pi[int(real_index)] = pi_strategic_deter[index]

    with open(output + '_config.json', "w") as fi:
        fi.write(js.dumps(dump_data(attr=attr, non_strategic=non_strategic_utility, strategic=u, components=num_components,
                                    strategic_deter=strategic_deterministic_utility, time=run_time, iterations=iterations,
                                    pi=pi, pi_non_strategic=non_strategic_pi, pi_strategic_deter=strategic_deter_pi,
                                    non_strategic_br=best_responses_non_strategic, strategic_br=best_responses,
                                    strategic_deter_br=best_responses_strategic_deterministic, alpha=alpha)))

    return attr

if __name__ == '__main__':
    experiment()
