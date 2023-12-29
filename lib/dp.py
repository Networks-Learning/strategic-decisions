import numpy as np
from lib import configuration_optimal as Configuration
import time
import json as js
import click

def dump_data(attr, utility, time, iter_num, pi, dp_br):
    out = {
        # Configuration
        "m": attr["m"],
        "seed": attr['seed'],
        "kappa": attr['kappa'],
        # Execution details
        "dp": utility,
        "time": time,
        "iterations": iter_num,
        "pi" : pi,
        "dp_br": dp_br,
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

# Computes the utility of a given policy and the best-response
# of the individuals of each feature value, constrained to the
# individuals with indexes i in [u,l).
def partial_utility(C, u, l, p, utility, policy):
    m=len(utility)
    best_responses=u*[None]+[i for i in range(u,m)]
    last_blocking=u
    util=utility[u]*p[u]*policy[u]
    best_responses[u]=u
    for i in range(u+1,l):
        if policy[i]==policy[i-1] and policy[i]>0:
            last_blocking=i
            util+=utility[i]*p[i]*policy[i]
            best_responses[i]=i
        elif C[i,last_blocking]<=policy[last_blocking]-policy[i]+(1e-9):
            util+=utility[last_blocking]*p[i]*policy[last_blocking]
            best_responses[i]=last_blocking
    return util,best_responses

# Performs one execution of the dynamic programming algorithm on a randomly
#  generated instance given the following parameters as command line arguments.
@click.command()
@click.option('--output', required=True, help="output directory")
@click.option('--m', default=4, help="Number of states")
@click.option('--seed', default=2, help="random number for seed.")
@click.option('--gamma', default=0.2, type=float, help="gamma parameter")
@click.option('--kappa', default=0.2, type=float, help="inverse sparsity of the graph")
@click.option('--population', default='normal', type=str, help="method of sampling population values")
@click.option('--cost_method', default='uniform', type=str, help="method of sampling cost values")
def experiment(output, m, seed, gamma, kappa, population, cost_method):
    attr = Configuration.generate_additive_configuration(
        m, seed, kappa=kappa, gamma=gamma, population=population, cost_method=cost_method)

    start = time.time()
    C=attr['C']
    p=attr['p']
    utility=attr['utility']
    cost_acc=np.zeros(m)
    cost_acc[0]=1
    for i in range(1,m):
        cost_acc[i]=cost_acc[i-1]-C[i,i-1]

    l_prime=m*[None]
    for j in range(m):
        cost_acc_new=cost_acc.copy()
        cost_acc_new+=1-cost_acc[j]
        for i in range(j+1,m):
            if cost_acc_new[i]<0 and cost_acc_new[i-1]>=0:
                l_prime[j]=i
                break
        if l_prime[j]==None:
            l_prime[j]=m

    # First non-positive state
    non_positive=m
    for i in range(m):
        if utility[i]<=0:
            non_positive=i
            break

    #Partial computation
    last_fixed=0
    total_policy=np.zeros(m)
    total_policy[0]=1
    iter_num=0
    while last_fixed!=m-1 and utility[last_fixed]>0:
        iter_num+=1
        choice={}
        trimmed={}
        top_pi=total_policy[last_fixed]
        U=np.zeros((m+1,m))
        best_responses={}
        pi_levels={}
        # Base Cases
        pi_levels[last_fixed+1,last_fixed]=np.zeros(m)
        trimmed[last_fixed+1,last_fixed]=m*[False]
        for u in range(last_fixed,non_positive-1):
            trimmed[non_positive-1,u]=m*[False]
            if C[non_positive-1,u]<=top_pi:
                policy=np.zeros(m)
                policy[u]=top_pi
                for i in range(u+1,non_positive-1):
                    policy[i]=policy[i-1]-C[i,i-1]
                pi_levels[non_positive-1,u]=policy
                
                policy[non_positive-1]=policy[non_positive-2]
                util_high,br_high=partial_utility(C,u,m,p,utility,policy)

                policy[non_positive-1]=policy[non_positive-2]-C[non_positive-1,non_positive-2]
                util_low,br_low=partial_utility(C,u,m,p,utility,policy)

                if util_high>util_low:
                    pi_levels[non_positive-1,u][non_positive-1]=policy[non_positive-2]
                    U[non_positive-1,u]=util_high
                    best_responses[non_positive-1,u]=br_high
                else:
                    pi_levels[non_positive-1,u][non_positive-1]=policy[non_positive-2]-C[non_positive-1,non_positive-2]
                    U[non_positive-1,u]=util_low
                    best_responses[non_positive-1,u]=br_low
            elif C[non_positive-2,u]<=top_pi:
                policy=np.zeros(m)
                policy[u]=top_pi
                for i in range(u+1,non_positive-1):
                    policy[i]=policy[i-1]-C[i,i-1]
                pi_levels[non_positive-1,u]=policy
                
                policy[non_positive-1]=policy[non_positive-2]
                util_high,br_high=partial_utility(C,u,m,p,utility,policy)
                pi_levels[non_positive-1,u][non_positive-1]=policy[non_positive-2]
                U[non_positive-1,u]=util_high
                best_responses[non_positive-1,u]=br_high
        
        for i in range(non_positive-2,last_fixed,-1):
            for u in range(i-1,last_fixed-1,-1):
                if C[i-1,u]<=top_pi:
                    trimmed[i,u]=m*[False]
                    best_responses[i,u]=u*[None]+[i for i in range(u,m)]
                    pi_levels[i,u]=np.zeros(m)
                    if C[i,u]<=top_pi:
                        upper_util=U[i+1,u]
                        
                        local_pi=top_pi-C[i-1,u]
                        #Fixing stair policy on the left of i
                        pi_levels[i,u][u]=top_pi
                        for j in range(u+1,i):
                            pi_levels[i,u][j]=pi_levels[i,u][j-1]-C[j,j-1]
                        rest_util,best_responses[i,u]=partial_utility(C,u,i,p,utility,pi_levels[i,u])

                        sub_best_responses=best_responses[i+1,i]    
                        sub_pi_levels=pi_levels[i+1,i].copy()
                        sub_pi_levels[i:]-=(top_pi-local_pi)
                        toTrim=np.logical_and(sub_pi_levels<0,sub_pi_levels>local_pi-top_pi)
                        sub_pi_levels[toTrim]=2
                        
                        last_not_trimmed=-1
                        for j in range(i,non_positive):
                            if sub_pi_levels[j]==2:
                                pi_levels[i,u][j]=sub_pi_levels[last_not_trimmed]
                            else:
                                last_not_trimmed=j
                                pi_levels[i,u][j]=sub_pi_levels[j]
                        
                        lower_util,lower_responses=partial_utility(C,i,m,p,utility,pi_levels[i,u])
                        trimmed[i,u]=trimmed[i+1,i].copy()
                        for j in range(i,m):
                            if lower_responses[j]!=sub_best_responses[j]:
                                trimmed[i,u][sub_best_responses[j]]=True
                                break

                        lower_util+=rest_util

                        best_responses[i,u][i:]=lower_responses[i:]
                        U[i,u]=lower_util
                        choice[i,u]='lower'

                        if lower_util<=upper_util:
                            U[i,u]=U[i+1,u]
                            choice[i,u]='higher'
                            trimmed[i,u]=trimmed[i+1,u].copy()
                            best_responses[i,u]=best_responses[i+1,u].copy()
                            pi_levels[i,u]=pi_levels[i+1,u].copy()
                    else:
                        local_pi=top_pi-C[i-1,u]
                        #Fixing stair policy on the left of i
                        pi_levels[i,u][u]=top_pi
                        for j in range(u+1,i):
                            pi_levels[i,u][j]=pi_levels[i,u][j-1]-C[j,j-1]
                        rest_util,best_responses[i,u]=partial_utility(C,u,i,p,utility,pi_levels[i,u])

                        sub_best_responses=best_responses[i+1,i]
                        sub_pi_levels=pi_levels[i+1,i].copy()
                        sub_pi_levels[i:]-=(top_pi-local_pi)
                        toTrim=np.logical_and(sub_pi_levels<0,sub_pi_levels>local_pi-top_pi)
                        sub_pi_levels[toTrim]=2
                        
                        last_not_trimmed=-1
                        for j in range(i,non_positive):
                            if sub_pi_levels[j]==2:
                                pi_levels[i,u][j]=sub_pi_levels[last_not_trimmed]
                            else:
                                last_not_trimmed=j
                                pi_levels[i,u][j]=sub_pi_levels[j]
                        
                        lower_util,lower_responses=partial_utility(C,i,m,p,utility,pi_levels[i,u])
                        trimmed[i,u]=trimmed[i+1,i].copy()
                        for j in range(i,m):
                            if lower_responses[j]!=sub_best_responses[j]:
                                trimmed[i,u][sub_best_responses[j]]=True
                                break

                        lower_util+=rest_util
                        best_responses[i,u][i:]=lower_responses[i:]
                        U[i,u]=lower_util
                        choice[i,u]='lower'
        temp_last_fixed=last_fixed
        for i in range(temp_last_fixed+1,m):
            if trimmed[temp_last_fixed+1,temp_last_fixed][i]==False:
                total_policy[i]=pi_levels[temp_last_fixed+1,temp_last_fixed][i]
                last_fixed=i
            else:
                break

    total_utility,total_responses=partial_utility(C,0,m,p,utility,total_policy)

    end = time.time()

    total_responses = {ind:int(x) for ind,x in enumerate(total_responses)}
    total_policy = {ind:float(x) for ind,x in enumerate(total_policy)}
    with open(output + '_config.json', "w") as fi:
        fi.write(js.dumps(dump_data(attr=attr, utility=total_utility, time=end - start, iter_num=iter_num,
                                pi=total_policy, dp_br=total_responses)))
    
    print("DP RunTime = " + str(end - start))
    print("Final Utility = " + str(total_utility))

if __name__ == '__main__':
    experiment()
    # experiment(output='test_dp', m=5, seed=545, gamma=0.0, kappa=0.25, population='uniform')

