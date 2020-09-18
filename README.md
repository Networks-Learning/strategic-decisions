# Title to be added here

This repository contains the code and data used in the papers [Decisions, Counterfactual Explanations and Strategic Behavior](https://arxiv.org/abs/2002.04333) and [Optimal Decision Making Under Strategic Behavior](https://arxiv.org/abs/1905.09239).

## Dependencies

All the experiments were performed using Python 3. In order to create a virtual environment and install the project dependencies you can run the following commands:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Code organization

The directory [lib](lib/) contains the source code of the algorithms described in the papers together with instance generators for synthetic and real data.

The directory [scripts](scripts/) contains bash scripts that use the aforementioned code and pass several parameters required for the various experiments.

The directory [notebooks](notebooks/) contains jupyter notebooks producing the figures appearing in the paper. Some notebooks use outputs produced by scripts and prior execution of some script is required. The required script can be found inside each notebook.

The directory [data](data/) contains the data used in the two papers.

The directory [figures](figures/) is used for saving the figures produced by the notebooks.

The directory [outputs](outputs/) is used for saving the text outputs produced by the scripts.

Each of the directories **scripts** and **notebooks** is consisted of two sub-directories named **counterfactuals** and **optimal** which contain paper-specific scripts/notebooks and they correspond to *Decisions, Counterfactual Explanations and Strategic Behavior* and *Optimal Decision Making Under Strategic Behavior* respectively. 

Each of the aforementioned directories contains self-explanatory README files whenever necessary.


## Citation

If you use parts of the code/data in this repository for your own research purposes, please consider citing:

    @software{strategic-decisions,
        author = {Tsirtsis, Stratis and Tabibian, Behzad and Khajehnejad, Moein and Singla, Adish and Sch{\"o}lkopf, Bernhard and Gomez-Rodriguez, Manuel},
        title = {TitleToBeAddedHere},
        url = {https://github.com/Networks-Learning/strategic-decisions/},
    }
