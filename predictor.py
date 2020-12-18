import pandas as pd
import networkx as nx
import numpy as np
import multiprocessing as mp
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def predict(features_file):
    data = pd.read_csv(features_file, sep="\t").rename(columns={'Unnamed: 0': 'K'})  # load data set
    data = data[(data.T != 0).any()]
    data = data.drop('K', axis=1)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data.drop('edge_sign', axis=1), data['edge_sign'],
                                                        test_size=0.10, random_state=101)

    logmodel = LogisticRegression()
    logmodel.fit(X_train, y_train)
    predictions = logmodel.predict(X_test)
    return classification_report(y_test, predictions)

if __name__ == "__main__":
    t1 = time_of_start_computation = datetime.now()

    with open('out-wiki.txt', 'w') as f:
        print(predict('./wiki-full-features'), file=f)

    t2 = datetime.now()
    triads_time = t2 - t1
    print(triads_time)
