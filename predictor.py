import pandas as pd
import networkx as nx
import numpy as np
import multiprocessing as mp
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def split(df):
    class_a = df[df.edge_sign == 0]
    class_b = df[df.edge_sign == 1]
    class_c = df[df.edge_sign == -1]
    return class_a, class_b, class_c

def split_balanced(df):
    nonedges, positives, negatives = split(df)
    test_size = int(len(df) * 0.01)
    test = pd.concat([nonedges.sample(test_size // 2), positives.sample(test_size // 2)])
    train = df.drop(test.index)

    X_test, Y_test = test.drop(['edge_sign'], axis=1), test.edge_sign
    X_train, Y_train = train.drop(['edge_sign'], axis=1), train.edge_sign

    return X_train, X_test, Y_train, Y_test

def predict(features_file):
    data = pd.read_csv(features_file, sep="\t").rename(columns={'Unnamed: 0': 'K'})  # load data set
    data = data[(data.T != 0).any()]
    data = data.drop('K', axis=1)

    X_train, X_test, Y_train, Y_test = split_balanced(data)
    logmodel = LogisticRegression(solver='sag', n_jobs=mp.cpu_count())
    logmodel.fit(X_train, y_train)
    predictions = logmodel.predict(X_test)
    return classification_report(y_test, predictions)

if __name__ == "__main__":
    print("var1")
    t1 = time_of_start_computation = datetime.now()

    with open('out-wiki-ml-varient1.tsv', 'w') as f:
        print(predict('./wiki-full-features'), file=f)

    t2 = datetime.now()
    triads_time = t2 - t1
    print(triads_time)


    print("var2")
    t1 = time_of_start_computation = datetime.now()
    with open('out-wiki-ml-varient2.tsv', 'w') as f:
        print(predict('./wiki-varient2-full-features'), file=f)

    t2 = datetime.now()
    triads_time = t2 - t1
    print(triads_time)

    print("var3")
    t1 = time_of_start_computation = datetime.now()
    with open('out-wiki-ml-varient3.tsv', 'w') as f:
        print(predict('./wiki-varient3-full-features'), file=f)

    t2 = datetime.now()
    triads_time = t2 - t1
    print(triads_time)

