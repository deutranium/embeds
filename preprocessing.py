import networkx as nx
import random
import utils as U


def train_test_val_split(g, train_size, val_size):
    """
    Split a given graph `g` into train, test and val splits based on their sizes.
    Arguments:
        g (NetworkX graph): graph to be split
        train_size (float): what fraction of edge count of `g` should go to the train split
        test_size (float): what fraction of edge count of `g` should go to the test split
        val_size (float): what fraction of edge count of `g` should go to the val split. Please note that validation edge set is a subset of train edge set.
    """
    g_edges = list(g.edges())
    g_edges_set = set(g_edges)
    g_nodes = list(g.nodes())
    l = len(g_edges)
    print(f"found {l} edges in graph")

    total_edge_count = len(g_edges)
    train_edge_count = int(total_edge_count * train_size)
    test_edge_count = total_edge_count - train_edge_count
    val_edge_count = int(total_edge_count * val_size)

    print("Starting sampling")

    train_positive_samples = random.sample(g_edges, train_edge_count)
    test_positive_samples = list(set(g_edges) - set(train_positive_samples))
    val_positive_samples = random.sample(train_positive_samples, val_edge_count)

    print(
        f"Sampled {len(train_positive_samples)}, {len(test_positive_samples)} and {len(val_positive_samples)} positive samples from train, test and val"
    )

    assert (
        len(train_positive_samples) == train_edge_count
    ), "Houston we have a problem with the size of positive train set"
    assert (
        len(test_positive_samples) == test_edge_count
    ), "Houston we have a problem with the size of positive test set"
    assert (
        len(val_positive_samples) == val_edge_count
    ), "Houston we have a problem with the size of positive val set"

    negative_samples = set()

    # ASSUMPTION: Number of edges in graph <= N^2/2 where N = number of nodes

    while len(negative_samples) < l:
        u, v = random.sample(g_nodes, 2)
        if ((u, v) in g_edges_set) or ((v, u) in g_edges_set):
            continue
        else:
            negative_samples.add((u, v))

    assert (
        len(negative_samples) == l
    ), "Houston we have a problem with the size of negative samples"

    train_negative_samples = random.sample(negative_samples, train_edge_count)
    test_negative_samples = list(set(negative_samples) - set(train_negative_samples))
    val_negative_samples = random.sample(train_negative_samples, val_edge_count)

    print(
        f"Sampled {len(train_negative_samples)}, {len(test_negative_samples)} and {len(val_negative_samples)} negative samples from train, test and val"
    )

    assert (
        len(train_negative_samples) == train_edge_count
    ), "Houston we have a problem with the size of negative train set"
    assert (
        len(test_negative_samples) == test_edge_count
    ), "Houston we have a problem with the size of negative test set"
    assert (
        len(val_negative_samples) == val_edge_count
    ), "Houston we have a problem with the size of negative val set"

    train_edges, train_labels = U.combine_samples(
        train_positive_samples, train_negative_samples
    )
    test_edges, test_labels = U.combine_samples(
        test_positive_samples, test_negative_samples
    )
    val_edges, val_labels = U.combine_samples(
        val_positive_samples, val_negative_samples
    )

    train_g = U.get_graph_from_edges(g, train_edges, train_labels)
    test_g = U.get_graph_from_edges(g, test_edges, test_labels)
    val_g = U.get_graph_from_edges(g, val_edges, val_labels)

    return (
        [train_g, train_edges, train_labels],
        [test_g, test_edges, test_labels],
        [val_g, val_edges, val_labels],
    )
