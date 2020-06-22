import numpy as np
import pandas as pd
import click
import ast

from lib.greedy_deter import compute_gd
from lib.greedy_rand import compute_gr
from lib.max_cover import compute_maxcov
from lib.min_cost import compute_mincos

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
@click.option('--k', type=int, required=True)
@click.option('--seed', type=int, default=1, help="random number for seed.")
@click.option('--leaking', cls=PythonLiteralOption, default='None', help="probabilities of sampling some other individual")
@click.option('--njobs', default=1, help="number of parallel threads")
@click.option('--output', required=True, help="output")
@click.option('--algo', required=True, help="algorithm to execute")
def experiment(data, output, seed, gamma, alpha, k, leaking, algo, njobs):
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
    if algo=="gd":
        compute_gd(output=output, C=c, U=u, Px=px, k=k, seed=seed, alpha=alpha, indexing=indexing, njobs=njobs)
    elif algo=="gr":
        compute_gr(output=output, C=c, U=u, Px=px, k=k, seed=seed, leaking=leaking, alpha=alpha, indexing=indexing, njobs=njobs)
    elif algo=="maxcov":
        compute_maxcov(output=output, C=c, U=u, Px=px, k=k, seed=seed, alpha=alpha, indexing=indexing, njobs=njobs)
    elif algo=="mincos":
        compute_mincos(output=output, C=c, U=u, Px=px, k=k, seed=seed, alpha=alpha, indexing=indexing, njobs=njobs)


if __name__ == '__main__':
    experiment()
