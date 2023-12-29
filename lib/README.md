# Code

The following tables contain a short description for each python file. **Common** refers to files related to both papers in this repository.The next two tables are paper-specific.

## Common

| Module                | Description                                                   |
|-----------------------|---------------------------------------------------------------|
| [credit_utils.py](credit_utils.py)  | Preprocesses the credit dataset. |
| [fico_utils.py](fico_utils.py)   | Preprocesses the lending dataset. |
| [utils.py](utils.py) | Contains auxiliary functions. |
| [real.py](lib/real.py) | Performs one experiment on real data with options about counterfactual explanations / full transparency. |

## Decisions, Counterfactual Explanations and Strategic Behavior

| Module                | Description                                                   |
|-----------------------|---------------------------------------------------------------|
| [fair.py](fair.py) | Performs one experiment on real data under a matroid constraint. |
| [min_cost.py](min_cost.py) | Finds minimum cost explanations. |
| [max_cover.py](max_cover.py) | Finds diverse explanations. |
| [greedy_deter.py](greedy_deter.py) | Finds explanations maximizing utility. |
| [greedy_rand.py](greedy_rand.py) | Finds explanations and a policy maximizing utility. |
| [greedy_fair.py](greedy_fair.py) | Finds explanations maximizing utility under a matroid constraint. |
| [configuration_counterfactuals.py](configuration_counterfactuals.py) | Contains instance generation functions. |

## Optimal Decision Making Under Strategic Behavior

| Module                | Description                                                   |
|-----------------------|---------------------------------------------------------------|
| [bruteforce.py](bruteforce.py) | Finds the optimal policy maximizing utility under full transparency. |
| [dp.py](dp.py) | Dynamic programming algorithm for finding a close to optimal policy on additive outcome monotonic instances. |
| [iterative.py](iterative.py) | Iterative algorithm for approximating the optimal policy. |
| [thres.py](thres.py) | Iterative algorithm that searches over all threshold policies and picks the one with maximum utility. |
| [configuration_optimal.py](configuration_optimal.py) | Contains instance generation functions. |