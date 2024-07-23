import torch
from torch_geometric.data import Data
def prepare_data():
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=torch.float)

    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 0, 3, 2, 2]], dtype=torch.long)

    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)

    train_mask = torch.tensor([True, True, False, False], dtype=torch.bool)
    test_mask = torch.tensor([False, False, True, True], dtype=torch.bool)
        
    data.train_mask = train_mask
    data.test_mask = test_mask

    return data