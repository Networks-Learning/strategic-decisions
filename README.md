# Decisions, Counterfactual Explanations and Strategic Behavior

This repository contains the code and data used in the paper *Decisions, Counterfactual Explanations and Strategic Behavior*. The full version can be found [here](https://arxiv.org/abs/2002.04333).

## Dependencies

All the experiments were performed using Python 3. In order to create a virtual environment and install the project dependencies you can run the following commands:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Code organization

The directory **lib** contains the source code of the algorithms described in the paper accompanied with instance generators for synthetic and real data.

The directory **scripts** contains bash scripts that use the aforementioned code and pass several parameters required for the various experiments.

The directory **notebooks** contains jupyter notebooks producing the figures appearing in the paper. Some notebooks use outputs produced by scripts and prior execution of some script is required. The required script can be found inside each notebook. Here, follows a matching between notebooks and figures with experimental results:

In the following tables, short descriptions of source code, notebooks and scripts are given.

| Module                | Description                                                   |
|-----------------------|---------------------------------------------------------------|
| [credit_utils.py](lib/credit_utils.py)  | Preprocesses the credit dataset. |
| [fico_utils.py](lib/fico_utils.py)   | Preprocesses the lending dataset. |
| [configuration.py](lib/configuration.py) | Contains instance generation functions. |
| [real.py](lib/real.py) | Performs one experiment on real data under a cardinality constraint. |
| [fair.py](lib/fair.py) | Performs one experiment on real data under a matroid constraint. |
| [min_cost.py](lib/min_cost.py) | Finds minimum cost explanations. |
| [max_cover.py](lib/max_cover.py) | Finds diverse explanations. |
| [greedy_deter.py](lib/greedy_deter.py) | Finds explanations maximizing utility. |
| [greedy_rand.py](lib/greedy_rand.py) | Finds explanations and a policy maximizing utility. |
| [greedy_fair.py](lib/greedy_fair.py) | Finds explanations maximizing utility under a matroid constraint. |
| [utils.py](lib/utils.py) | Contains auxiliary functions. |

| Script                | Description                                                   |
|-----------------------|---------------------------------------------------------------|
| [credit.sh](scripts/credit.sh)  | Generates the credit dataset. |
| [fico.sh](scripts/fico.sh)  | Generates the lending dataset. |
| [alphas.sh](scripts/alphas.sh)  | Performs experiments on real data for various values of alpha. |
| [real.sh](scripts/real.sh)  | Performs experiments on real data for various values of k. |
| [syncomp.sh](scripts/syncomp.sh)  | Performs experiments on synthetic data for various values of cost, m and k . |
| [fair.sh](scripts/fair.sh)  | Performs experiments on real data with a cardinality and a matroid constraint. |

| Notebook              | Description                                                   |
|-----------------------|---------------------------------------------------------------|
| [alphas.ipynb](notebooks/alphas.ipynb)  | Produces Figure 1. |
| [real.ipynb](notebooks/real.ipynb)     | Produces Figure 2. |
| [fair.ipynb](notebooks/fair.ipynb)     | Produces Figure 3. |
| [syncomp.ipynb](notebooks/syncomp.ipynb)     | Produces Figure 6. |
| [viz.ipynb](notebooks/viz.ipynb)     | Produces Figure 7 and Table 2. |


## Citation

If you use parts of the code in this repository for your own research purposes, please consider citing:

    @article{tsirtsis2020decisions,
        title={Decisions, Counterfactual Explanations and Strategic Behavior},
        author={Tsirtsis, Stratis and Gomez-Rodriguez, Manuel},
        journal={arXiv preprint arXiv:2002.04333},
        year={2020}
    }