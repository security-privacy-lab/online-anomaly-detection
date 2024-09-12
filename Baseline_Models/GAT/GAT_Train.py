import torch
import torch.optim as optim
import torch.nn.functional as F
from GAT_Model import GAT
# from GAT_Demo_Data import prepare_data
from sklearn.metrics import f1_score, roc_auc_score


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

    _, pred = out[data.train_mask].max(dim=1)
    f1 = f1_score(data.y[data.train_mask].cpu(), pred.cpu(), average='weighted')
    return loss.item(), f1

def test(model, data):
    model.eval()
    out = model(data)
    _, pred = model(data).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / int(data.test_mask.sum())
    f1 = f1_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu(), average='weighted')
    auc = roc_auc_score(data.y[data.test_mask].cpu().detach().numpy(), out[data.test_mask].cpu().detach().numpy()[:, 1], average='weighted', multi_class='ovr')
    return acc, f1, auc

def main():
    data = prepare_data()
    model = GAT(in_channels=data.num_features, out_channels=2)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(200):
        loss, train_f1 = train(model, data, optimizer, criterion)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}, Train F1: {train_f1:.4f}')

    test_acc, test_f1, test_auc = test(model, data)
    print(f'Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test AUC: {test_auc:.4f}')

if __name__ == "__main__":
    main()
