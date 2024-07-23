import torch
import torch.optim as optim
import torch.nn.functional as F
from GAT_Demo_Model import GAT
from GAT_Demo_Data import prepare_data

def prepare_data():
    data = torch.load("graph_creation/processed_data/gat_data.pt")
    return data

def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data):
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / int(data.test_mask.sum())
    return acc

def main():
    data = prepare_data()
    model = GAT(in_channels=data.num_features, out_channels=2)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(200):
        loss = train(model, data, optimizer, criterion)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}')

    test_acc = test(model, data)
    print(f'Test Accuracy: {test_acc:.4f}')

if __name__ == "__main__":
    main()
