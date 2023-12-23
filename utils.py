import numpy as np
import random
import networkx as nx
import helper as H
import pandas as pd

# from stellargraph.data import EdgeSplitter
from sklearn.model_selection import train_test_split


def get_graph_from_edges(g, edges, labels):
    g = nx.Graph(g)
    nodes = g.nodes()

    res = nx.Graph()
    res.add_nodes_from(nodes)

    positive_edges = [tuple(edges[i]) for i, val in enumerate(edges) if labels[i]]
    temp = len(positive_edges)
    res.add_edges_from(positive_edges)
    positive_edges = set(positive_edges)

    # print(positive_edges)
    # print(f"{len(positive_edges)} positive edges detected from {temp} edges")
    return res


def read_graph_from_path(path):
    return nx.read_edgelist(path, nodetype=int, create_using=nx.Graph(), delimiter=",")


# def get_splits(g, test_size=0.2, val_size=0.1, seed=1, method="global"):
#     abs_train_size = 1-test_size
#     val_size = val_size/abs_train_size
#     train_size = 1-val_size

#     test_splitter = EdgeSplitter(g)
#     not_test_g, test_examples, test_labels = test_splitter.train_test_split(p=test_size, seed=seed, method=method)
#     test_g = get_graph_from_edges(g, test_examples, test_labels)

#     # TRAIN AND GRAPH
#     train_splitter = EdgeSplitter(not_test_g, g)
#     not_train_g, examples, labels = train_splitter.train_test_split(p=0.9999999, seed=seed, method=method)
#     train_g = get_graph_from_edges(g, examples, labels)

#     # train and val split
#     train_examples, val_examples, train_labels, val_labels = train_test_split(examples, labels, train_size=train_size, test_size=val_size)

#     print(f"""INFO: Split made with {len([i for i in train_labels if i==1])}, {len([i for i in val_labels if i==1])} and {len(test_g.edges())} edges in train, val and test respectively.
#     `train_graph` and `test_graph` have {len(train_g.edges())} and {len(test_g.edges())} edges respectively whereas the original graph had {len(g.edges())} edges""")


#     return test_g, test_examples, test_labels, train_g, train_examples, train_labels, val_examples, val_labels

# class GetBestCLF():
#     def __init__(self, examples_train, examples_val, labels_train, labels_val, embedding_train) -> None:
#         results = [H.run_link_prediction(op, examples_train, examples_val, labels_train, labels_val, embedding_train) for op in H.binary_operators]
#         self.results = results
#         best_result = max(results, key=lambda result: result["score"])
#         self.best_result = best_result

#         print(f"Best result from '{best_result['binary_operator'].__name__}'")

#         self.df = pd.DataFrame(
#             [(result["binary_operator"].__name__, result["score"]) for result in results],
#             columns=("name", "ROC AUC score"),
#         ).set_index("name")


# def evaluate_stuff(best_clf, examples_test, labels_test, embedding_test, best_bo):
#     test_score = H.evaluate_link_prediction_model(best_clf,
#         examples_test,
#         labels_test,
#         embedding_test,
#         best_bo
#     )
#     print(
#         f"ROC AUC score on test set using '{best_bo.__name__}': {test_score}"
#     )

#     return test_score


def combine_samples(positive, negative):
    l = len(positive)
    Y = [1] * l + [0] * l
    X = positive + negative

    temp = list(zip(X, Y))
    random.shuffle(temp)

    X, Y = zip(*temp)
    return X, Y


def get_edge_embeds(edge_list, get_model_embed):
    embeds = []
    for edge in edge_list:
        u, v = edge
        this_embed = list(np.multiply(get_model_embed(u), get_model_embed(v)))
        embeds.append(this_embed)

    return embeds


def get_node_embeds(node_list, get_model_embed):
    embeds = []
    for node in node_list:
        this_embed = list(get_model_embed(node))
        embeds.append(this_embed)

    return embeds
