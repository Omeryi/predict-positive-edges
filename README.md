# predict-positive-edges

## Files in this project
* `normalize_wiki_file.py` - creates a tsv file representing a directed and signed graph based on Wikipedia's elections' data.
* `graph.py` - contains some utils relating to graph and creates three variants of the original graph, writing them to a tsv file.
* `features.py` - calculates the features, which are the number of every triads by type for every pair of nodes in the dataset.
* `predictor.py` - conducts a 10-fold cross validation where every fold contains a balanced amount of node-pairs connected by a positive edge and non-connected nodes of the same embeddedness.
* `Predicting Positive Links in an Online Social Network.pdf` - research paper. 

* `datasets/` - contains the results of `normalize_wiki_files.py` and `graph.py`
* `calculated_features/` - contains the results of `features.py`
* `predictions/` - contains the results of `predictor.py`
