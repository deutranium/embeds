import pickle
import wandb
import json
import multiprocessing
from stellargraph import StellarGraph, datasets
import networkx as nx
import utils as U
import embeddings as E
import preprocessing as P
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


def get_dataset():
    ds = datasets.Cora()
    print(ds.description)
    g, _ = ds.load(largest_connected_component_only=True, str_node_ids=True)
    nx_g = StellarGraph.to_networkx(g)
    nx_g = nx.Graph(nx_g)
    return nx_g


nx_g = get_dataset()


ITER = 1
WORKERS = multiprocessing.cpu_count()
SPLIT_SEED = 1

wandb.init()


def main(path=None):
    train = 0.8
    test = 0.2
    val = 0.1

    print(
        f"Running train, test and val splits with values {train}, {test} and {val} respectively"
    )

    with open(path, 'rb') as f:
        (
        [train_g, train_edges, train_labels],
        [test_g, test_edges, test_labels],
        [val_g, val_edges, val_labels],
        ) = pickle.load(f)

    p = wandb.config.p
    q = wandb.config.q
    num_walks = wandb.config.NUM_WALKS
    w_len = wandb.config.W_LEN
    dimensions = wandb.config.DIMENSIONS
    window_size = wandb.config.WINDOW_SIZE

    n2v = E.BetterNode2Vec(
        train_g,
        p=p,
        q=q,
        num_walks=num_walks,
        w_len=w_len,
        dimensions=dimensions,
        window_size=window_size,
        workers=WORKERS,
        iter=ITER,
    )

    # get edge embeddings
    train_embeds_n2v = U.get_edge_embeds(train_edges, n2v.get_embedding)
    test_embeds_n2v = U.get_edge_embeds(test_edges, n2v.get_embedding)
    val_embeds_n2v = U.get_edge_embeds(val_edges, n2v.get_embedding)

    edge_clf_n2v = LogisticRegression()
    edge_clf_n2v.fit(train_embeds_n2v, train_labels)

    val_preds_n2v = edge_clf_n2v.predict_proba(val_embeds_n2v)[:, 1]
    val_roc_n2v = roc_auc_score(val_labels, val_preds_n2v)
    val_ap_n2v = average_precision_score(val_labels, val_preds_n2v)

    test_preds_n2v = edge_clf_n2v.predict_proba(test_embeds_n2v)[:, 1]
    test_roc_n2v = roc_auc_score(test_labels, test_preds_n2v)
    test_ap_n2v = average_precision_score(test_labels, test_preds_n2v)

    wandb.log(
        {
            "val_roc": val_roc_n2v,
            "val_ap": val_ap_n2v,
            "test_roc": test_roc_n2v,
            "test_ap": test_ap_n2v,
            "roc_auc_score": test_roc_n2v,
        }
    )


if __name__ == "__main__":
    main(path="./dataset_splits/cora/undirected.pickle")
