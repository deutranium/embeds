import pickle
import wandb
import multiprocessing
import utils as U
import embeddings as E
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score

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

    node_clf_nbne = LogisticRegression(multi_class="multinomial", max_iter=500)
    node_clf_nbne.fit(train_node_embeds_nbne, train_Y)

    val_preds_nbne = node_clf_nbne.predict(val_node_embeds_nbne)
    val_score = node_clf_nbne.score(val_node_embeds_nbne, val_Y)
    val_f1 = f1_score(val_preds_nbne, val_Y, average="macro")
    # val_roc_nbne = roc_auc_score(val_Y, val_preds_nbne, multi_class="ovo")
    # val_ap_nbne = average_precision_score(val_Y, val_preds_nbne)

    test_preds_nbne = node_clf_nbne.predict(test_node_embeds_nbne)
    test_score = node_clf_nbne.score(test_node_embeds_nbne, test_Y)
    test_f1 = f1_score(test_preds_nbne, test_Y, average="macro")
    # test_roc_nbne = roc_auc_score(test_Y, test_preds_nbne, multi_class="ovo")
    # test_ap_nbne = average_precision_score(test_Y, test_preds_nbne)

    wandb.log(
        {
            "val_score": val_score,
            "test_score": test_score,
            "val_f1_score": val_f1,
            "test_f1_score": test_f1
        }
    )


if __name__ == "__main__":
    main(path="./label_classification/dataset_splits/pubmed_nodes/undirected.pickle")
