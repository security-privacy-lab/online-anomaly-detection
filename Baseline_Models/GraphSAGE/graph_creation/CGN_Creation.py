import os
import torch
from torch_geometric.data import DataLoader, Dataset
from torch_geometric.utils import to_undirected
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.optim as optim

class CustomGraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(CustomGraphDataset, self).__init__(root, transform, pre_transform)
        self.graph_paths = self._get_graph_paths(root)

    def _get_graph_paths(self, root):
        graph_paths = []
        for subdir in os.listdir(root):
            subdir_path = os.path.join(root, subdir)
            if os.path.isdir(subdir_path):
                data_path = os.path.join(subdir_path, "data.pt")
                if os.path.exists(data_path):
                    graph_paths.append(data_path)
        return graph_paths

    def len(self):
        return len(self.graph_paths)  # Return the number of graphs

    def get(self, idx):
        data = torch.load(self.graph_paths[idx])
        data.edge_index = to_undirected(data.edge_index)
        return data

train_dataset = CustomGraphDataset("graph_creation/processed_data/fold_1/train/")
test_dataset = CustomGraphDataset("graph_creation/processed_data/fold_1/test/")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Example data to get input dimensions
example_data = train_dataset.get(0)
input_dim = example_data.num_features
output_dim = example_data.y.max().item() + 1

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

model = GCN(input_dim=input_dim, hidden_dim=64, output_dim=output_dim)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    total_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        print(f"Batch {batch_idx}: Output shape: {output.shape}, Data.y shape: {data.y.shape}")  # Debugging statement
        loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
        pred = output.argmax(dim=1)
        correct += (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
    return correct / len(loader.dataset)

for epoch in range(1, 201):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
