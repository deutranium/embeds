import numpy as np
import pickle
import wandb
import json
import multiprocessing
# from stellargraph import StellarGraph, datasets
import networkx as nx
import utils as U
import embeddings as E
# import preprocessing as P
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
# from sklearn.metrics import average_precision_score

ITER = 1
WORKERS = multiprocessing.cpu_count()
SPLIT_SEED = 1

# wandb.init()


def main(seed, path=None):
    print("SEED", seed)

    splits_path = f"./dataset_splits/cora/cora_splits_{seed}.json"

    train = 0.8
    test = 0.2
    val = 0.1

    print(
        f"Running train, test and val splits with values {train}, {test} and {val} respectively"
    )

    with open(path, "rb") as f:
        g = pickle.load(f)

    

    with open(splits_path) as f:
        [[train_X, train_Y],
            [val_X, val_Y],
            [test_X, test_Y]] = json.load(f)

    all_nodes = list(g.nodes())

    dimensions = 128
    win_size = 5
    num_permutations = 8
    thresh_len = 12
    threshold = 4
    k = 3
    # print("DIMENSIONS:", dimensions)

    print("VALUES:", dimensions, win_size, num_permutations, thresh_len, threshold, k)

    nbne = E.NBNE(
        g=g,
        k=k,
        num_permutations=num_permutations,
        thresh_len=thresh_len,
        threshold=threshold,
        dimensions=dimensions,
        window_size=win_size,
        workers=WORKERS,
        iter=ITER,
    )
    

    # get edge embeddings
    train_node_embeds_nbne = U.get_node_embeds(train_X, nbne.get_embedding)
    val_node_embeds_nbne = U.get_node_embeds(val_X, nbne.get_embedding)
    test_node_embeds_nbne = U.get_node_embeds(test_X, nbne.get_embedding)

    node_clf_nbne = LogisticRegression(multi_class="multinomial")
    node_clf_nbne.fit(train_node_embeds_nbne, train_Y)

    val_preds_nbne = node_clf_nbne.predict(val_node_embeds_nbne)
    val_score = node_clf_nbne.score(val_node_embeds_nbne, val_Y)

    acc = accuracy_score(val_Y, val_preds_nbne)
    f1_micro = f1_score(val_Y, val_preds_nbne, average="micro")
    f1_macro = f1_score(val_Y, val_preds_nbne, average="macro")

    print("accuracy", acc)
    print("f1 micro", f1_micro)
    print("f1 macro", f1_macro)

    # with open('train_cora_embeds.txt', 'w') as f:
    #     f.write(train_node_embeds_nbne)

    # for node, embeds in zip(all_nodes, all_embeds):
    #     this_str = node + " "
    #     e = [str(i) for i in embeds]
    #     this_str += " ".join(e)
    #     this_str.strip()
    #     this_str += "\n"
    #     with open('cora_64.embeddings', 'a') as the_file:
    #         the_file.write(this_str)



    # print(train_node_embeds_nbne)
    # print(np.array(train_node_embeds_nbne).shape)



    # edge_clf_nbne = LogisticRegression()
    # edge_clf_nbne.fit(train_embeds_nbne, train_labels)

    # val_preds_nbne = edge_clf_nbne.predict_proba(val_embeds_nbne)[:, 1]
    # val_roc_nbne = roc_auc_score(val_labels, val_preds_nbne)
    # val_ap_nbne = average_precision_score(val_labels, val_preds_nbne)

    # test_preds_nbne = edge_clf_nbne.predict_proba(test_embeds_nbne)[:, 1]
    # test_roc_nbne = roc_auc_score(test_labels, test_preds_nbne)
    # test_ap_nbne = average_precision_score(test_labels, test_preds_nbne)

    # wandb.log(
    #     {
    #         "val_roc": val_roc_nbne,
    #         "val_ap": val_ap_nbne,
    #         "test_roc": test_roc_nbne,
    #         "test_ap": test_ap_nbne,
    #         "roc_auc_score": test_roc_nbne,
    #     }
    # )


if __name__ == "__main__":
    main(seed="11", path="./dataset_splits/cora/cora_graph.pickle")
