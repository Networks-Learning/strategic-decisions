import numpy as np

from lib import configuration as Configuration
import time
import click
import json as js

def dump_data(attr, utility, time, pi, bruteforce_br):
    out = {
        # Configuration
        "m": attr["m"],
        "seed": attr['seed'],
        "sparsity": attr['degree_of_sparsity'],
        "kappa": attr['kappa'],
        # Execution details
        "bruteforce": utility,
        "time": time,
        "pi" : pi,
        "bruteforce_br": bruteforce_br,
    }
    return out


# Computes the utility of a given policy and the best-response
# of the individuals of each feature value.
def compute_utility(pi_c, p, C, utility):
    m = pi_c.size
    u = 0
    br=np.zeros(m,dtype=int)
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

# Recursively checks all the possible combinations of best-responses
# returning the one with maximum utility among the feasible ones. 
def backtrack(matched, pi, p, utility, C, m, best_utility_till_now):
    if len(matched) == m:
        return LP(matched, pi, p, utility, C, best_utility_till_now)

    best_utility = -1
    best_pi = None
    temp_matched = [a for a in matched]
    ii = len(matched)
    candidate_values = np.where(C[ii, :] <= np.min(C[ii, :]) + 1)[0]
    for j in candidate_values:
        matched.append(j)
        [temp_utility, temp_pi] = backtrack(matched, pi, p, utility, C, m, best_utility_till_now)
        if best_utility < temp_utility:
            best_utility = temp_utility
            best_pi = temp_pi
        if best_utility_till_now < temp_utility:
            best_utility_till_now = temp_utility
        matched = [a for a in temp_matched]
    return [best_utility, best_pi]

# Checks if the given individuals' best-responses are feasible and computes
# the optimal policy based on them, returning the corresponding utility.
def LP(matched, pi, p, utility, C, best_utility_till_now):

    fancy_pi = np.ones([len(pi)])
    fancy_pi[utility < 0] = 0
    if np.sum(p * fancy_pi[matched] * utility[matched]) < best_utility_till_now:
        return [-np.Inf, None]

    m = pi.size

    pi_new=np.zeros(m)
    pi_new[0]=1
    problem=False
    for i in range(1,m):
        upper_bound=1
        for k in range(m):
            k_star=matched[k]
            if k_star<i:
                upper_bound=min(pi_new[k_star]+C[k,i]-C[k,k_star],upper_bound)
        if upper_bound<0:
            problem=True
            break
        else:
            if utility[i]>0:
                pi_new[i]=upper_bound
            else:
                pi_new[i]=0
    u,br=compute_utility(pi_new, p, C, utility)
    if not np.array_equal(br,matched):
        problem=True

    if problem==True:
        return [-np.Inf, None]
    else:
        return [u,pi_new]

# Performs one execution of the bruteforce algorithm on a randomly generated
# instance given the following parameters as command line arguments.
@click.command()
@click.option('--output', required=True, help="output directory")
@click.option('--m', default=4, help="Number of states")
@click.option('--seed', default=2, help="random number for seed.")
@click.option('--sparsity', default=2, help="sparsity of the graph")
@click.option('--kappa', default=0.2, type=float, help="inverse sparsity of the graph")
@click.option('--gamma', default=0.2, type=float, help="gamma parameter")
@click.option('--additive', is_flag=True, default=False, help="if used, it generates additive configuration")
def experiment(output, m, seed, sparsity, kappa, gamma, additive):
    if additive:
        attr = Configuration.generate_additive_configuration(
            m, seed, kappa=kappa, gamma=gamma)
    else:
        attr = Configuration.generate_pi_configuration(
            m, seed, accepted_percentage=1, degree_of_sparsity=sparsity, gamma=gamma)
        attr["pi"] = np.zeros(m)
    
    matched = []
    start = time.time()
    [best_utility, best_pi] = backtrack(
        matched, attr["pi"], attr["p"], attr["utility"], attr["C"], attr["m"], -np.Inf)

    best_utility, br = compute_utility(best_pi,attr['p'],attr['C'],attr['utility'])
    end = time.time()
    print("Bruteforce RunTime = " + str(end - start))
    print("Final Utility = " + str(best_utility))

    br = {ind:int(x) for ind,x in enumerate(br)}
    best_pi = {ind:float(x) for ind,x in enumerate(best_pi)}
    with open(output + '_config.json', "w") as fi:
        fi.write(js.dumps(dump_data(attr=attr, utility=best_utility, time=end - start,
                            pi=best_pi, bruteforce_br=br)))


if __name__ == '__main__':
    experiment()
