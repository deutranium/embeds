import pickle
import wandb
import json
import multiprocessing
# from stellargraph import StellarGraph, datasets
import networkx as nx
import utils as U
import embeddings as E
import preprocessing as P
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


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
    train_embeds_nbne = U.get_edge_embeds(train_edges, n2v.get_embedding)
    test_embeds_nbne = U.get_edge_embeds(test_edges, n2v.get_embedding)
    val_embeds_nbne = U.get_edge_embeds(val_edges, n2v.get_embedding)

    edge_clf_nbne = LogisticRegression()
    edge_clf_nbne.fit(train_embeds_nbne, train_labels)

    val_preds_nbne = edge_clf_nbne.predict_proba(val_embeds_nbne)[:, 1]
    val_roc_nbne = roc_auc_score(val_labels, val_preds_nbne)
    val_ap_nbne = average_precision_score(val_labels, val_preds_nbne)

    test_preds_nbne = edge_clf_nbne.predict_proba(test_embeds_nbne)[:, 1]
    test_roc_nbne = roc_auc_score(test_labels, test_preds_nbne)
    test_ap_nbne = average_precision_score(test_labels, test_preds_nbne)

    wandb.log(
        {
            "val_roc": val_roc_nbne,
            "val_ap": val_ap_nbne,
            "test_roc": test_roc_nbne,
            "test_ap": test_ap_nbne,
            "roc_auc_score": test_roc_nbne,
        }
    )


if __name__ == "__main__":
    main(path="./dataset_splits/astrophysics/directed.pickle")
