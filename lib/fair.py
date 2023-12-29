import numpy as np
import pandas as pd
import click

from lib.greedy_deter import compute_gd
from lib.greedy_fair import compute_gf

@click.command()
@click.option('--data', required=True)
@click.option('--gamma', type=float, required=True)
@click.option('--alpha',type=float ,required=True)
@click.option('--k', type=int, required=True)
@click.option('--njobs', default=1, help="number of parallel threads")
@click.option('--seed', default=1, help="random seed")
@click.option('--output', required=True, help="output")
@click.option('--cost_method', type=str, help="method of setting the cost function")
def experiment(data, output, gamma, seed, alpha, k, njobs, cost_method):
    """
    Executes the greedy deterministic algorithm on real data, with a matroid constraint.

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
    njobs : int
        number of parallel threads to be used
    output : string
        output directory prefix (e.g., outputs/exec1_)
    """

    # Read outcomes
    u = pd.read_csv(data+'_pyx.csv', index_col=0, names=["ID", "Probability"], header=0, dtype={'Probability': float})
    u.sort_values(by=["Probability"],inplace=True, ascending=False)
    indexing=u.index.values.flatten().tolist()
    u=u.values.flatten()-gamma

    # Read costs
    cost=pd.read_csv(data+'_cost_' + str(cost_method) + '.csv', index_col=0, header=0)
    cost.columns = cost.columns.astype(int)
    cost=cost[indexing]
    c = cost.reindex(indexing).to_numpy()
    c = c*alpha # scaling

    # Read population
    px_df = pd.read_csv(data+'_px.csv', index_col=0, header=0)
    px = px_df.reindex(indexing).to_numpy().flatten()

    # Read natural vectors and set the partition matroid
    natural_vectors = pd.read_csv(data+'_vectors.csv', index_col=0, header=0)
    partitions = natural_vectors['Age group'].reindex(indexing).astype(int).to_numpy().flatten()
    num_of_partitions = len(np.unique(partitions))
    
    # Compute
    compute_gf(num_of_partitions=num_of_partitions, partitions=partitions, output=output, C=c, U=u, Px=px, k=k,
                seed=seed, alpha=alpha, indexing=indexing, njobs=njobs)


if __name__ == '__main__':
    experiment()
