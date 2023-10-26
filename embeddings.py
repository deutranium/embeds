import json
import networkx as nx
import random
from gensim.models import Word2Vec
from stellargraph.data import BiasedRandomWalk
from stellargraph import StellarGraph
import pickle
import node2vec


class NBNE:
    def __init__(
        self,
        g,
        k,
        num_permutations,
        thresh_len,
        threshold,
        dimensions,
        window_size,
        workers,
        iter,
        min_thresh=1,
        seed=1,
    ) -> None:
        self.g = g
        self.k = k
        self.num_permutations = num_permutations
        self.thresh_len = thresh_len
        self.thresh = threshold
        # QUESTION: thresh < k always right?
        self.min_thresh = min_thresh
        self.directed = nx.is_directed(self.g)

        # for Word2Vec
        self.dimensions = dimensions
        self.window_size = window_size
        self.workers = workers
        self.iter = iter

        self.SEED = seed

        self.walks = self.get_sentences()
        self.model = self.get_model()

    def get_sentences(self):
        g = self.g
        k = self.k
        permutations = self.num_permutations

        nodes = list(g.nodes())
        sentences = []

        for p in range(permutations):
            print(f"PERMUTATION: {p}")

            for node in nodes:
                neighbours = list(nx.all_neighbors(g, node))
                random.shuffle(neighbours)
                len_n = len(neighbours)

                if len_n:
                    for i in range(0, len_n, k):
                        start = min(i, max(0, len_n - k))
                        this_sentence = [node] + neighbours[start : start + k]

                        node_thresh = min(self.thresh, g.degree(node))

                        for t in range(self.min_thresh, node_thresh + 1):
                            this_threshold_sentence = self.get_threshold_rw(
                                this_sentence, t
                            )
                            sentences.append(this_threshold_sentence)
                else:
                    sentences.append([node])

        return sentences

    def get_threshold_rw(self, nodes, t):
        """
        Get a random walk for a list of `nodes` and a threshold `t`
        """

        threshold_count = {node: t for node in nodes}
        activated = {node: 0 for node in nodes}

        idx = 0

        while idx < len(activated) <= self.thresh_len:
            k = list(activated.keys())
            this_node = k[idx]

            this_neighbours = list(self.g.neighbors(this_node))
            random.shuffle(this_neighbours)

            for neighbour in this_neighbours:
                if neighbour in threshold_count:
                    threshold_count[neighbour] += 1
                else:
                    threshold_count[neighbour] = 1

                if threshold_count[neighbour] == t:
                    activated[neighbour] = 0

            idx += 1
        return list(activated.keys())

    def get_model(self):
        """
        Get Work2Vec model
        """

        model = Word2Vec(
            self.walks,
            vector_size=self.dimensions,
            window=self.window_size,
            min_count=0,
            sg=1,
            workers=self.workers,
            epochs=self.iter,
            seed=self.SEED,
        )
        return model

    def get_embedding(self, u):
        idx = self.model.wv.key_to_index[u]
        return self.model.wv[idx]

    def store_embeddings(self, path):
        d = self.model.wv.key_to_index
        embeds = self.model.wv

        with open(f"{path}_d_thresh.json", "w", encoding="utf-8") as f:
            json.dump(d, f, indent=4)
        with open(f"{path}_embeds_thresh.pickle", "wb") as handle:
            pickle.dump(embeds, handle, protocol=pickle.HIGHEST_PROTOCOL)


class BetterNode2Vec:
    def __init__(
        self, g, p, q, num_walks, w_len, dimensions, window_size, workers, iter, seed=1
    ) -> None:
        self.g = g
        self.p = p
        self.q = q
        self.num_walks = num_walks
        self.w_len = w_len

        # for Word2Vec
        self.dimensions = dimensions
        self.window_size = window_size
        self.workers = workers
        self.iter = iter

        self.SEED = seed

        self.walks = self.get_sentences()
        self.model = self.get_model()

    def get_sentences(self):
        g, p, q, num_walks, w_len = self.g, self.p, self.q, self.num_walks, self.w_len
        self.n2v = node2vec.Node2Vec(
            g, walk_length=w_len, num_walks=num_walks, p=p, q=q, seed=self.SEED
        )
        return self.n2v.walks

    def get_model(self):
        """
        Get Word2Vec model
        """

        model = Word2Vec(
            self.walks,
            vector_size=self.dimensions,
            window=self.window_size,
            min_count=0,
            sg=1,
            workers=self.workers,
            epochs=self.iter,
            seed=self.SEED,
        )

        return model

    def get_embedding(self, u):
        idx = self.model.wv.key_to_index[u]
        return self.model.wv[idx]

    def store_embeddings(self, path):
        d = self.model.wv.key_to_index
        embeds = self.model.wv

        with open(f"{path}_d_n2v.json", "w", encoding="utf-8") as f:
            json.dump(d, f, indent=4)
        with open(f"{path}_embeds_n2v.pickle", "wb") as handle:
            pickle.dump(embeds, handle, protocol=pickle.HIGHEST_PROTOCOL)


class Node2Vec:
    def __init__(
        self, g, p, q, num_walks, rw_len, dimensions, window_size, workers, iter
    ) -> None:
        g = StellarGraph.from_networkx(g)

        self.g = g
        self.p = p
        self.q = q
        self.num_walks = num_walks
        self.rw_len = rw_len

        # for Word2Vec
        self.dimensions = dimensions
        self.window_size = window_size
        self.workers = workers
        self.iter = iter

        self.walks = self.get_sentences()
        self.model = self.get_model()

    def get_sentences(self):
        g = self.g
        p, q = self.p, self.q

        rw = BiasedRandomWalk(g)
        sentences = rw.run(g.nodes(), n=self.num_walks, length=self.rw_len, p=p, q=q)
        return sentences

    def get_model(self):
        """
        Get Word2Vec model
        """

        model = Word2Vec(
            self.walks,
            vector_size=self.dimensions,
            window=self.window_size,
            min_count=0,
            sg=1,
            workers=self.workers,
            epochs=self.iter,
        )

        return model

    def get_embedding(self, u):
        idx = self.model.wv.key_to_index[u]
        return self.model.wv[idx]

    def store_embeddings(self, path):
        d = self.model.wv.key_to_index
        embeds = self.model.wv

        with open(f"{path}_d_n2v.json", "w", encoding="utf-8") as f:
            json.dump(d, f, indent=4)
        with open(f"{path}_embeds_n2v.pickle", "wb") as handle:
            pickle.dump(embeds, handle, protocol=pickle.HIGHEST_PROTOCOL)
