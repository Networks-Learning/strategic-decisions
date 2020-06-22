import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from scipy import stats
import click
import pickle
import os

from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

@click.command()
@click.option('--data', required=True, help="data file")
@click.option('--output', required=True, help="output file prefix")
@click.option('--njobs', default=1, type=int, help="number of parallel threads")
def experiment(data, output, njobs):

    np.random.seed(seed=42)
    # Read data
    original_df = pd.read_csv(data)

    # Clean data (make sure each sample belongs to one age group)
    cleaned_df = original_df.drop('Single', axis=1)
    cleaned_df = cleaned_df.drop('HistoryOfOverduePayments', axis=1)
    def clean_ages(row):
        age_columns = ['Age_lt_25', 'Age_in_25_to_40', 'Age_in_40_to_59', 'Age_geq_60']
        age_groups = row[age_columns]==1
        candidate_columns = age_groups[age_groups].index.tolist()
        random_age_group = np.random.choice(candidate_columns, size=1)[0]
        row[age_columns] = 0
        row[random_age_group] = 1
        return row

    print('Cleaning...')
    cleaned_df = cleaned_df.apply(lambda x: clean_ages(x), axis=1)

    # Turn to arrays and scale
    y = cleaned_df['NoDefaultNextMonth'].to_numpy()
    X = cleaned_df.drop('NoDefaultNextMonth', axis=1).to_numpy()

    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)

    # Try several classifiers and numbers of clusters
    print('Cross validation...')
    X_numerical = X_scaled[:,6:]
    opt_acc = 0
    k_values = [5, 10, 20, 50, 100, 200]
    for k in k_values:

        # Split data to k clusters based on their numerical features
        print(' '.join(['Clustering with k',str(k)]))
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X_numerical)

        # Replace numerical features with one-hot encoding of the respective cluster
        enc = preprocessing.OneHotEncoder(sparse=False)
        cats = enc.fit_transform(kmeans.labels_.reshape(-1,1))

        X_summ = np.zeros((X_scaled.shape[0], 6+k))
        for ind, x in enumerate(X_scaled):
            X_summ[ind,:6] = X_scaled[ind,:6]
            X_summ[ind,6:] = cats[ind]

        # Train a Multi-Layer Perceptron
        print(' '.join(['MLP with k',str(k)]))
        mlp_clf = MLPClassifier(random_state=42, max_iter=500)
        mlp_scores = cross_val_score(mlp_clf, X_summ, y, cv=5, n_jobs=5, verbose=0)
        acc = np.mean(mlp_scores)
        if acc > opt_acc:
            opt_acc = acc
            opt_k_clf = (k, MLPClassifier(random_state=42, max_iter=500))

        # Train a Support Vector Machine
        print(' '.join(['SVM with k',str(k)]))
        svm_clf = svm.SVC(random_state=42)
        svm_scores = cross_val_score(svm_clf, X_summ, y, cv=5, n_jobs=5, verbose=0)
        acc = np.mean(svm_scores)
        if acc > opt_acc:
            opt_acc = acc
            opt_k_clf = (k, svm.SVC(random_state=42))

        # Train a Logistic Regression Classifier
        print(' '.join(['LR with k',str(k)]))
        lr_clf = LogisticRegression(random_state=42, max_iter=500)
        lr_scores = cross_val_score(lr_clf, X_summ, y, cv=5, n_jobs=5, verbose=0)
        acc = np.mean(lr_scores)
        if acc > opt_acc:
            opt_acc = acc
            opt_k_clf = (k, LogisticRegression(random_state=42, max_iter=500))

        # Train a Decision Tree
        print(' '.join(['DT with k',str(k)]))
        dt_clf = DecisionTreeClassifier(random_state=42)
        dt_scores = cross_val_score(dt_clf, X_summ, y, cv=5, n_jobs=5, verbose=0)
        acc = np.mean(dt_scores)
        if acc > opt_acc:
            opt_acc = acc
            opt_k_clf = (k, DecisionTreeClassifier(random_state=42))


    # opt_k, opt_clf, opt_acc = (100, LogisticRegression(random_state=42, max_iter=500), 0.8042) # HELPER
    opt_k, opt_clf = opt_k_clf
    print('Optimal accuracy: ' + str(opt_acc))
    print('Optimal classifier: ' + str(opt_clf))
    print('Optimal k: ' + str(opt_k))
    print('Total samples: ' + str(X_scaled.shape[0]))
    
    # Data representation depending on optimal k
    kmeans = KMeans(n_clusters=opt_k, random_state=42).fit(X_numerical)
    enc = preprocessing.OneHotEncoder(sparse=False)
    cats = enc.fit_transform(kmeans.labels_.reshape(-1,1))
    X_summ = np.zeros((X_scaled.shape[0], 6+opt_k))
    for ind, x in enumerate(X_scaled):
        X_summ[ind,:6] = X_scaled[ind,:6]
        X_summ[ind,6:] = cats[ind]

    if os.path.isfile(output + '_clf.pk'):
        # Load classifier if already trained
        with open(output + '_clf.pk', 'rb') as f:
            opt_clf = pickle.load(f)
    else:
        # Retrain optimal classifier
        opt_clf.fit(X_summ, y)
        with open(output + '_clf.pk', 'wb') as f:
            pickle.dump(opt_clf, f)


    print('Organizing in groups...')
    feature_groups = {}
    for married in [0,1]:
        for age_group in range(4):
            for education in range(4):
                for cluster_id in range(opt_k):
                    feature_groups[(married, age_group, education, cluster_id)]={}
                    
                    # Recreate group vector (in training form)
                    vector = np.zeros(6+opt_k)
                    vector[0] = married
                    vector[1+age_group] = 1
                    vector[5] = education/3
                    vector[6+cluster_id] = 1
                    
                    # Get P(y|x)
                    prob = opt_clf.predict_proba(vector.reshape(1,-1))[0][1]
                    
                    # Recreate group  vector (in natural form)                
                    natural_vector = np.zeros(3+9)
                    natural_vector[0] = married
                    natural_vector[1] = age_group
                    natural_vector[2] = education
                    cluster_center = kmeans.cluster_centers_[cluster_id]
                    cluster_center = np.array([max(0,x) for x in cluster_center])
                    cluster_center = np.array([min(1,x) for x in cluster_center])
                    temp_vector = np.concatenate([np.zeros(6), cluster_center])
                    inverse_vector = np.rint(min_max_scaler.inverse_transform(temp_vector.reshape(1,-1))[0])
                    
                    # sanity check
                    if inverse_vector[-2] == 0:
                        inverse_vector[-1] = 0
                    
                    natural_vector[3:] = inverse_vector[6:]
                    feature_groups[(married, age_group, education, cluster_id)]['Probability'] = prob
                    feature_groups[(married, age_group, education, cluster_id)]['Population'] = 0
                    feature_groups[(married, age_group, education, cluster_id)]['Natural vector'] = natural_vector


    # Compute P(x)
    for ind, x in enumerate(X_summ):
        married = int(x[0])
        for i in range(4):
            if x[1+i]==1:
                age_group=i
        education = int(3*x[5])
        cluster_id = kmeans.labels_[ind]
        feature_groups[(married, age_group, education, cluster_id)]['Population'] += 1

    for i_group_id in list(feature_groups):
        feature_groups[i_group_id]['Population'] /= X_summ.shape[0]

    # Compute gamma (50th percentile of all individual P(y|x) values -- half population accepted by threshold)
    probs = [(feature_groups[group_id]['Probability'],feature_groups[group_id]['Population']) for group_id in feature_groups]
    probs = sorted(probs, key=lambda  x: x[0])
    cumulative_population = 0
    for prob, pop in probs:
        if cumulative_population>=0.5:
            gamma=prob
            break
        else:
            cumulative_population+=pop


    # Compute cost function
    m = len(feature_groups)
    cost = np.full((m,m), fill_value=2.0) # set unreachable states' cost to 2 (>1)
    centroids = np.array([x['Natural vector'] for x in list(feature_groups.values())])
    centroid_cost = np.full((opt_k, opt_k), 2.0)

    # Cost depending on the numerical values
    for i_cluster in range(opt_k):
        for j_cluster in range(opt_k):
            i_vector = feature_groups[(0,0,0,i_cluster)]['Natural vector']
            j_vector = feature_groups[(0,0,0,j_cluster)]['Natural vector']
            # History of overdue payments can only increase
            if i_vector[-2] <= j_vector[-2] and i_vector[-1] <= j_vector[-1]:
                # Maximum percentile shift among all numerical features
                max_percentile = -1
                for k in range(3,12):
                    i_percentile = stats.percentileofscore(centroids[:,k], i_vector[k])/100
                    j_percentile = stats.percentileofscore(centroids[:,k], j_vector[k])/100
                    if np.abs(i_percentile - j_percentile) > max_percentile:
                        max_percentile = np.abs(i_percentile - j_percentile)
                centroid_cost[i_cluster, j_cluster] = max_percentile
    
    # Pairwise cost depending on categorical features and cluster id
    for i_group, i_group_id in enumerate(list(feature_groups)):
        for j_group, j_group_id in enumerate(list(feature_groups)):
            i_vector = feature_groups[(i_group_id[0],i_group_id[1],i_group_id[2],i_group_id[3])]['Natural vector']
            j_vector = feature_groups[(j_group_id[0],j_group_id[1],j_group_id[2],j_group_id[3])]['Natural vector']
            # Marriage, Age, Education not actionable
            if (i_vector[:3]==j_vector[:3]).all():
                cost[i_group, j_group] = centroid_cost[i_group_id[3], j_group_id[3]]

    # Store summary
    with open(output+'_summary.txt','w') as f:
        f.write('Optimal accuracy: ' + str(opt_acc) + '\n')
        f.write('Optimal classifier: ' + str(opt_clf) + '\n')
        f.write('Optimal k: ' + str(opt_k) + '\n')
        f.write('Total samples: ' + str(X_scaled.shape[0]) + '\n')
        f.write('Gamma: ' + str(gamma) + '\n')

    # Store pairwise costs 
    with open(output+'_cost.csv','w') as f:
        f.write(',')
        f.write(','.join([str(x) for x in range(m)]))
        f.write('\n')
        for i in range(m):
            f.write(str(i)+',')
            f.write(','.join(cost[i].astype(str).tolist()))
            f.write('\n')

    # Store population
    with open(output+'_px.csv', 'w') as f:
        f.write('ID,Population\n')
        for i_group, i_group_id in enumerate(list(feature_groups)):
            f.write(str(i_group)+','+str(feature_groups[i_group_id]['Population'])+'\n')

    # Store P(y|x)
    with open(output+'_pyx.csv', 'w') as f:
        f.write('ID,Probability\n')
        for i_group, i_group_id in enumerate(list(feature_groups)):
            f.write(str(i_group)+','+str(feature_groups[i_group_id]['Probability'])+'\n')

    # Store feature vectors
    vectors_df = pd.DataFrame(columns=['Married', 'Age group', 'Education', 'MaxBillAmountOverLast6Months', 
                                        'MaxPaymentAmountOverLast6Months', 'MonthsWithZeroBalanceOverLast6Months',
                                        'MonthsWithLowSpendingOverLast6Months', 'MonthsWithHighSpendingOverLast6Months',
                                        'MostRecentBillAmount', 'MostRecentPaymentAmount', 'TotalOverdueCounts',
                                        'TotalMonthsOverdue'])
    for i_group, i_group_id in enumerate(list(feature_groups)):
        vectors_df = vectors_df.append(pd.Series(feature_groups[i_group_id]['Natural vector'].tolist(), index=vectors_df.columns), ignore_index=True)
    vectors_df.to_csv(output+'_vectors.csv')

if __name__ == '__main__':
    experiment()
