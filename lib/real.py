from itertools import combinations_with_replacement
import numpy as np
import pandas as pd
import click
import ast

from lib.greedy_deter import compute_gd
from lib.greedy_rand import compute_gr
from lib.max_cover import compute_maxcov
from lib.min_cost import compute_mincos
from lib.iterative import compute_iter
from lib.thres import compute_thres

class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)

@click.command()
@click.option('--data', required=True)
@click.option('--gamma', type=float, required=True)
@click.option('--alpha',type=float ,required=True)
@click.option('--output', required=True, help="output")
@click.option('--algo', required=True, help="algorithm to execute")
@click.option('--k', type=int, default=2)
@click.option('--seed', type=int, default=1, help="random number for seed.")
@click.option('--leaking', cls=PythonLiteralOption, default='None', help="probabilities of sampling some other individual")
@click.option('--max_iter', type=int, default=100, help="max iterations for the iterative algorithm")
@click.option('--cost_method', type=str, help="method of setting the cost function")
@click.option('--njobs', default=1, help="number of parallel threads")
@click.option('--noisy_pyx', default=0.0, type=float, help="relative variance of noise for predictions")
@click.option('--noisy_cost',default=0.0, type=float, help="relative variance of noise for costs")
@click.option('--confounding',default=0.0, type=float, help="level of confounding")
def experiment(data, output, seed, gamma, alpha, k, leaking, algo, cost_method, njobs, max_iter, noisy_pyx, noisy_cost, confounding):
    """
    Executes one of the algorithms on real data.

    Parameters
    ----------
    data : string 
        data directory prefix (e.g., data/processed/fico)
    output : string
        output directory prefix (e.g., outputs/exec1_)
    seed : int
        random seed for reproducibility
    gamma : float
        gamma parameter value
    alpha : float
        alpha parameter value
    k : int
        maximum number of explanations
    leaking : list of floats
        leakage probabilities
    algo : string
        algorithm to be used
    cost_method : string
        method of setting the cost function (euclidean or max percentile shift)
    njobs : int
        number of parallel threads to be used
    max_iter : int
        max iterations for the iterative algorithm
    noisy_pyx : float
        relative variance of noise for predictions
    noisy_cost : float
        relative variance of noise for costs
    confounding : float
        level of confounding
    """

    # Read outcomes
    u_temp = pd.read_csv(data+'_pyx.csv', index_col=0, names=["ID", "Probability"], header=0, dtype={'Probability': float})
    if noisy_pyx==0.0:
        u = u_temp.copy()
    else:
        np.random.seed(seed)
        u = u_temp.copy()
        pyx_std = u_temp['Probability'].std()
        u['Probability'] = u_temp['Probability'] + np.random.normal(0, noisy_pyx*pyx_std, len(u_temp['Probability']))
        u['Probability'] = u['Probability'].apply(lambda x : max(0,min(x,1)))

    u.sort_values(by=["Probability"],inplace=True, ascending=False)
    indexing=u.index.values.flatten().tolist()
    u=u.values.flatten()-gamma

    u_temp=u_temp.reindex(indexing)
    u_temp=u_temp.values.flatten()-gamma

    # Read costs
    cost_temp=pd.read_csv(data+'_cost_' + str(cost_method) + '.csv', index_col=0, header=0)
    cost_temp.columns = cost_temp.columns.astype(int)
    if noisy_cost==0.0:
        cost = cost_temp.copy()
    else:
        np.random.seed(seed)
        cost = cost_temp.copy()
        cost_std = np.std([x for x in cost.to_numpy().flatten() if x<1.01])
        cost = cost + np.random.normal(0, noisy_cost*cost_std, cost.shape)
        cost = cost.applymap(lambda x : max(0,x))
        for i in range(len(cost.index)):
            for j in range(len(cost.columns)):
                if i==j:
                    cost.loc[i,j] = 0
    
    cost=cost[indexing]
    c = cost.reindex(indexing).to_numpy()
    c = c*alpha # scaling

    cost_temp=cost_temp[indexing]
    c_temp = cost_temp.reindex(indexing).to_numpy()
    c_temp = c_temp*alpha # scaling

    # Read population
    px_df = pd.read_csv(data+'_px.csv', index_col=0, header=0)
    px = px_df.reindex(indexing).to_numpy().flatten()

    # Compute
    if noisy_cost==0.0 and noisy_pyx==0.0:
        if algo=="gd": # Greedy deterministic
            compute_gd(output=output, C=c, U=u, Px=px, k=k, seed=seed, alpha=alpha, indexing=indexing, njobs=njobs)
        elif algo=="gr": # Greedy randomized
            compute_gr(output=output, C=c, U=u, Px=px, k=k, seed=seed, leaking=leaking, alpha=alpha, indexing=indexing, njobs=njobs)
        elif algo=="maxcov": # Maximum coverage
            compute_maxcov(output=output, C=c, U=u, Px=px, k=k, seed=seed, alpha=alpha, indexing=indexing, njobs=njobs)
        elif algo=="mincos": # Minimum cost
            compute_mincos(output=output, C=c, U=u, Px=px, k=k, seed=seed, alpha=alpha, indexing=indexing, njobs=njobs)
        elif algo=="it": # Iterative
            compute_iter(output=output, C=c, U=u, Px=px, seed=seed, alpha=alpha, indexing=indexing, max_iter=max_iter, verbose=False, njobs=njobs, split_components=False)
        elif algo=="itc": # Iterative (with components)
            compute_iter(output=output, C=c, U=u, Px=px, seed=seed, alpha=alpha, indexing=indexing, max_iter=max_iter, verbose=False, njobs=njobs, split_components=True, confounding=confounding)
        elif algo=="th": # Threshold
            compute_thres(output=output, C=c, U=u, Px=px, seed=seed, alpha=alpha, indexing=indexing, njobs=njobs, confounding=confounding)
    else:
        if algo=="itc": # Iterative (with components)
            compute_iter(output=output, C=c, U=u, Px=px, seed=seed, alpha=alpha, indexing=indexing, max_iter=max_iter, verbose=False, njobs=njobs, split_components=True, U_real=u_temp, C_real=c_temp)
        elif algo=="th": # Threshold
            compute_thres(output=output, C=c, U=u, Px=px, seed=seed, alpha=alpha, indexing=indexing, njobs=njobs, U_real=u_temp, C_real=c_temp)

# TODO: Make sure to give cost_method as argument when calling real in counterfactuals' scripts
            
if __name__ == '__main__':
    experiment()
