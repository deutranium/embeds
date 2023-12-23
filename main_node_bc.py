import pickle
import wandb
import multiprocessing
import utils as U
import embeddings as E
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
# from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
import numpy as np

ITER = 1
WORKERS = multiprocessing.cpu_count()
SPLIT_SEED = 1

wandb.init()


def main(path=None):
    train = 0.8
    test = 0.2
    val = 0.1

    print(
        f"Running train, test and val splits with values {train}, {test} and \
        {val} respectively"
    )

    with open(path, "rb") as f:
        (
            g,
            [train_X, train_Y],
            [val_X, val_Y],
            [test_X, test_Y],
        ) = pickle.load(f)

    dimensions = wandb.config.DIMENSIONS
    win_size = wandb.config.WINDOW_SIZE
    num_permutations = wandb.config.NUM_PERMUTATIONS
    thresh_len = wandb.config.THRESH_LEN
    threshold = wandb.config.THRESHOLD
    k = wandb.config.k


    if k + 1 < threshold:
        wandb.log(
            {
                "val_roc": -1,
                "val_ap": -1,
                "test_roc": -1,
                "test_ap": -1,
                "roc_auc_score": -1,
            }
        )
        return 0


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

    node_clf_nbne = OneVsRestClassifier(estimator= LogisticRegression(max_iter=300))
    node_clf_nbne.fit(train_node_embeds_nbne, train_Y)

    val_preds_nbne = node_clf_nbne.predict(val_node_embeds_nbne)
    val_score = node_clf_nbne.score(val_node_embeds_nbne, val_Y)
    val_f1 = f1_score(val_preds_nbne, val_Y, average="macro")
    val_f1_micro = f1_score(val_preds_nbne, val_Y, average="micro")
    # val_roc_nbne = roc_auc_score(val_Y, val_preds_nbne, multi_class="ovo")
    # val_ap_nbne = average_precision_score(val_Y, val_preds_nbne)

    test_preds_nbne = node_clf_nbne.predict(test_node_embeds_nbne)
    test_score = node_clf_nbne.score(test_node_embeds_nbne, test_Y)
    test_f1 = f1_score(test_preds_nbne, test_Y, average="macro")
    test_f1_micro = f1_score(test_preds_nbne, test_Y, average="micro")
    # test_roc_nbne = roc_auc_score(test_Y, test_preds_nbne, multi_class="ovo")
    # test_ap_nbne = average_precision_score(test_Y, test_preds_nbne)

    wandb.log(
        {
            "val_score": val_score,
            "test_score": test_score,
            "val_f1_score": val_f1,
            "test_f1_score": test_f1,
            "val_f1_score_micro": val_f1_micro,
            "test_f1_score_micro": test_f1_micro
        }
    )


if __name__ == "__main__":
    main(path="./label_classification/dataset_splits/bc_nodes/undirected.pickle")
