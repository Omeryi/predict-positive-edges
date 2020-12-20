import pandas as pd
import networkx as nx
import numpy as np
import multiprocessing as mp
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def get_safe_balanced_split(target, trainSize=0.9, getTestIndexes=True, shuffle=False, seed=None):
    classes, counts = np.unique(target, return_counts=True)
    nPerClass = float(len(target))*float(trainSize)/float(len(classes))
    if nPerClass > np.min(counts):
        print("Insufficient data to produce a balanced training data split.")
        print("Classes found %s"%classes)
        print("Classes count %s"%counts)
        ts = float(trainSize*np.min(counts)*len(classes)) / float(len(target))
        print("trainSize is reset from %s to %s"%(trainSize, ts))
        trainSize = ts
        nPerClass = float(len(target))*float(trainSize)/float(len(classes))
    # get number of classes
    nPerClass = int(nPerClass)
    print("Data splitting on %i classes and returning %i per class"%(len(classes),nPerClass ))
    # get indexes
    trainIndexes = []
    for c in classes:
        if seed is not None:
            np.random.seed(seed)
        cIdxs = np.where(target==c)[0]
        cIdxs = np.random.choice(cIdxs, nPerClass, replace=False)
        trainIndexes.extend(cIdxs)
    # get test indexes
    testIndexes = None
    if getTestIndexes:
        testIndexes = list(set(range(len(target))) - set(trainIndexes))
    # shuffle
    if shuffle:
        trainIndexes = random.shuffle(trainIndexes)
        if testIndexes is not None:
            testIndexes = random.shuffle(testIndexes)
    # return indexes
    return trainIndexes, testIndexes

# TODO: EDIT FUNCTION SO IT IS SAFE
def split_balanced(data, target, test_size=0.1):

    classes = np.unique(target)
    # can give test_size as fraction of input data size of number of samples
    if test_size<1:
        n_test = np.round(len(target)*test_size)
    else:
        n_test = test_size
    n_train = max(0, len(target)-n_test)
    n_train_per_class = max(1, int(np.floor(n_train/len(classes))))
    n_test_per_class = max(1, int(np.floor(n_test/len(classes))))

    ixs = []
    for cl in classes:
        if (n_train_per_class+n_test_per_class) > np.sum(target==cl):
            # if data has too few samples for this class, do upsampling
            # split the data to training and testing before sampling so data points won't be
            #  shared among training and test data
            splitix = int(np.ceil(n_train_per_class/(n_train_per_class+n_test_per_class)*np.sum(target==cl)))
            ixs.append(np.r_[np.random.choice(np.nonzero(target==cl)[0][:splitix], n_train_per_class),
                np.random.choice(np.nonzero(target==cl)[0][splitix:], n_test_per_class)])
        else:
            ixs.append(np.random.choice(np.nonzero(target==cl)[0], n_train_per_class+n_test_per_class,
                replace=False))

    # take same num of samples from all classes
    ix_train = np.concatenate([x[:n_train_per_class] for x in ixs])
    ix_test = np.concatenate([x[n_train_per_class:(n_train_per_class+n_test_per_class)] for x in ixs])

    X_train = data[ix_train,:]
    X_test = data[ix_test,:]
    y_train = target[ix_train]
    y_test = target[ix_test]

    return X_train, X_test, y_train, y_test

def predict(features_file):
    data = pd.read_csv(features_file, sep="\t").rename(columns={'Unnamed: 0': 'K'})  # load data set
    data = data[(data.T != 0).any()]
    data = data.drop('K', axis=1)

    X_train, X_test, Y_train, Y_test = split_balanced(data, data['edge_sign'])
    logmodel = LogisticRegression(solver='sag', n_jobs=mp.cpu_count())
    logmodel.fit(X_train, y_train)
    predictions = logmodel.predict(X_test)
    return classification_report(y_test, predictions)

if __name__ == "__main__":
    t1 = time_of_start_computation = datetime.now()

    with open('out-wiki-test-manipulation.txt', 'w') as f:
        print(predict('./wiki-full-features'), file=f)

    t2 = datetime.now()
    triads_time = t2 - t1
    print(triads_time)
