# Processed data

Files with the prefix *credit* contain data for the credit dataset while files with the prefix *fico* contain data for the lending dataset. 

The files \**dataset*\*_clf.pk contain the classifier trained to estimate P(y|x).

The files \**dataset*\*_cost.csv contain an array of cost values for adapting from a feature value with ID i to a feature value with ID j.

The files \**dataset*\*_px.csv contain estimates about the population distribution P(x) for each feature value.

The files \**dataset*\*_pyx.csv contain estimates about the probabilities P(y=1|x) for each feature value.

The files \**dataset*\*_vectors.csv contain the real features corresponding to each feature value.

The files \**dataset*\*_summary.txt contain extra information about preprocessing.
