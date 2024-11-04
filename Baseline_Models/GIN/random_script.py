import json
import numpy as np


def func():
    embeddings = np.array(
        list(json.load(open("GNN_node_embeddings.json", "r"))))

    print(embeddings.shape)


if __name__ == "__main__":
    func()
