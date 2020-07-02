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
# from lib.greedy_heur import compute_heur_compon

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
@click.option('--njobs', default=1, help="number of parallel threads")
def experiment(data, output, seed, gamma, alpha, k, leaking, algo, njobs, max_iter):
    """
    Executes one of the algorithms on real data.

    Parameters
    ----------
    data : string 
        data directory prefix (e.g., data/processed/fico)
    gamma : float
        gamma parameter value
    alpha : float
        alpha parameter value
    k : int
        maximum number of explanations
    seed : int
        random seed for reproducibility
    leaking : list of floats
        leakage probabilities
    njobs : int
        number of parallel threads to be used
    output : string
        output directory prefix (e.g., outputs/exec1_)
    algo : string
        algorithm to be used
    """

    # Read outcomes
    u = pd.read_csv(data+'_pyx.csv', index_col=0, names=["ID", "Probability"], header=0, dtype={'Probability': np.float})
    u.sort_values(by=["Probability"],inplace=True, ascending=False)
    indexing=u.index.values.flatten().tolist()
    u=u.values.flatten()-gamma

    # Read costs
    cost=pd.read_csv(data+'_cost.csv', index_col=0, header=0)
    cost.columns = cost.columns.astype(int)
    cost=cost[indexing]
    c = cost.reindex(indexing).to_numpy()
    c = c*alpha # scaling

    # Read population
    px_df = pd.read_csv(data+'_px.csv', index_col=0, header=0)
    px = px_df.reindex(indexing).to_numpy().flatten()

    # Compute
    if algo=="gd": # Greedy deterministic
        compute_gd(output=output, C=c, U=u, Px=px, k=k, seed=seed, alpha=alpha, indexing=indexing, njobs=njobs)
    elif algo=="gr": # Greedy randomized
        compute_gr(output=output, C=c, U=u, Px=px, k=k, seed=seed, leaking=leaking, alpha=alpha, indexing=indexing, njobs=njobs)
    elif algo=="maxcov": # Maximum coverage
        compute_maxcov(output=output, C=c, U=u, Px=px, k=k, seed=seed, alpha=alpha, indexing=indexing, njobs=njobs)
    elif algo=="mincos": # Minimum cost
        compute_mincos(output=output, C=c, U=u, Px=px, k=k, seed=seed, alpha=alpha, indexing=indexing, njobs=njobs)
    elif algo=="it": # Iterative
        compute_iter(output=output, C=c, U=u, Px=px, seed=seed, alpha=alpha, indexing=indexing, max_iter=max_iter, verbose=False, njobs=njobs)
    elif algo=="th": # Shifted threshold
        compute_thres(output=output, C=c, U=u, Px=px, seed=seed, alpha=alpha, indexing=indexing, njobs=njobs)
    # elif algo=="gh": # Greedy heuristic
    #     compute_heur_compon(output=output, C=c, U=u, Px=px, seed=seed, alpha=alpha, indexing=indexing, njobs=njobs)

if __name__ == '__main__':
    experiment()
