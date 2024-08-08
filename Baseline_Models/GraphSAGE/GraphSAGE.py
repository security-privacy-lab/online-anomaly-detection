import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import DataLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_networkx
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


'''GraphSAGE:
A fixed size sample of its neighbors is selected which helps to limit the computational complexity and memory usage, as only
a subset of the neighbors is considered at each layer 
Can be used to be mean, sum, max-pooling, LSTM-based and others
Aggregation: is taken care by SAVECong -> aggregates features from the neighbors using a mean-based aggregation by default
Combining Node and neibor features are done by SAGEConv
Multi-Layer Stacking -> self.conv
Training : uses supervised loss: Cross EntropyLoss. Updates the parameter of the nueral network to minimize the loss
Final layer's output for each node can be used as node's embedding. Used for node classifcation. 
'''
# Define configuration parameters
class Args:
    model = 'graphsage_mean'  # Model type
    learning_rate = 0.01  # Initial learning rate
    model_size = 'small'  # Model size (small or large)
    train_prefix = ''  # Prefix identifying training data
    epochs = 20  # Number of training epochs
    dropout = 0.5  # Dropout rate
    weight_decay = 5e-4  # Weight decay (L2 regularization)
    max_degree = 128  # Maximum node degree
    samples_1 = 25  # Number of samples in layer 1
    samples_2 = 10  # Number of samples in layer 2
    samples_3 = 0  # Number of samples in layer 3 (if applicable)
    dim_1 = 128  # Output dimension size for layer 1
    dim_2 = 128  # Output dimension size for layer 2
    random_context = True  # Whether to use random context or direct edges
    batch_size = 128  # Batch size
    sigmoid = False  # Whether to use sigmoid loss
    identity_dim = 0  # Dimension of identity embedding features (if positive)
    base_log_dir = '.'  # Base directory for logging and saving embeddings
    validate_iter = 500  # Validation iteration frequency
    validate_batch_size = 256  # Validation batch size
    gpu = 0  # GPU index to use
    print_every = 5  # Frequency of printing training info
    max_total_steps = 10**10  # Maximum number of training steps

# Instantiate configuration parameters
FLAGS = Args()
# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)  
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# Define the GraphSAGE Node model
class GraphSAGENodeModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GraphSAGENodeModel, self).__init__()
        # Initialize a list of convolutional layers
        self.convs = nn.ModuleList()  
        # First layer
        self.convs.append(SAGEConv(input_dim, hidden_dim)) 
        # Additional layers 
        for _ in range(num_layers - 1):  
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            # Final fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)  
    
    def forward(self, data):
        # Get node features and edge indices
        x, edge_index = data.x, data.edge_index  
        for conv in self.convs:
            # Apply convolution and ReLU activation
            x = conv(x, edge_index).relu()  
            # Apply dropout
            x = F.dropout(x, p=FLAGS.dropout, training=self.training)  
        # Store node embeddings from the last layer    
        node_embeddings = x  
        # Apply final fully connected layer
        out = self.fc(node_embeddings)  
        # Return output and node embeddings
        return out, node_embeddings  

# Function to hash IP addresses
def hash_ip(ip):
    # Convert IP to an integer
    return int.from_bytes(ip.encode(), 'little') % (10 ** 8)  

# Read and preprocess data(Change the directory here)
df = pd.read_csv("C:/Users/seoji/Downloads/Darknet.CSV")  # Load dataset
# Convert timestamps
df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %I:%M:%S %p")  
# Create minute-level timestamp
df["timestamp_minute"] = df["Timestamp"].dt.floor("min") 
# Sort by timestamp 
df = df.sort_values(by=["Timestamp"])  
# Group by minute
grouped = df.groupby("timestamp_minute")  

# Prepare graphs
pyg_graphs = []
# Initialize feature scaler
scaler = StandardScaler()  
all_features = []

for timestamp_minute, group in grouped:
    index = 0
    # Initialize directed graph
    G = nx.DiGraph()  
    labels = []
    previous_backward_node = None

    for idx, row in group.iterrows():
         # Hash source IP
        src_ip = hash_ip(row["Src IP"]) 
        # Hash destination IP
        dst_ip = hash_ip(row["Dst IP"])  

        # Define forward traffic node features
        forward_features = {
            "total_pkts": row.get("Total Fwd Packet", 0),
            "pkts_per_s": row.get("Fwd Packets/s", 0),
            "iat_std": row.get("Fwd IAT Std", 0),
            "total_len_pkt": row.get("Total Length of Fwd Packet", 0),
            "pkt_len_std": row.get("Fwd Packet Length Std", 0),
            "seg_size_avg": row.get("Fwd Segment Size Avg", 0),
            "init_win_bytes": row.get("FWD Init Win Bytes", 0),
            "pkt_len_mean": row.get("Fwd Packet Length Mean", 0),
            "iat_max": row.get("Fwd IAT Max", 0),
            "avg_pkt_size": row.get("Average Packet Size", 0),
            "subflow_bytes": row.get("Subflow Fwd Bytes", 0),
            "ip": src_ip,
            "direction": 0,
        }

        forward_node_id = f"{index}_fwd"
        # Add forward node
        G.add_node(forward_node_id, **forward_features)  
        # Collect features
        all_features.append(list(forward_features.values()))  
        # Determine label
        forward_label = 0 if row["Label"].lower() in ["nonvpn", "non-tor"] else 1  
        labels.append(forward_label)  
        # Collect label

        # Define backward traffic node features (negated)
        backward_features = {
            "total_pkts": -row.get("Total Bwd packets", 0),
            "pkts_per_s": -row.get("Bwd Packets/s", 0),
            "iat_std": -row.get("Bwd IAT Std", 0),
            "total_len_pkt": -row.get("Total Length of Bwd Packet", 0),
            "pkt_len_std": -row.get("Bwd Packet Length Std", 0),
            "seg_size_avg": -row.get("Bwd Segment Size Avg", 0),
            "init_win_bytes": -row.get("Bwd Init Win Bytes", 0),
            "pkt_len_mean": -row.get("Bwd Packet Length Mean", 0),
            "iat_max": -row.get("Bwd IAT Max", 0),
            "avg_pkt_size": -row.get("Average Packet Size", 0),
            "subflow_bytes": -row.get("Subflow Bwd Bytes", 0),
            "ip": dst_ip,
            "direction": -1,
        }

        backward_node_id = f"{index}_bkd"
        # Add backward node
        G.add_node(backward_node_id, **backward_features)  
         # Collect features
        all_features.append(list(backward_features.values())) 

        backward_label = 0 if row["Label"].lower() in ["nonvpn", "non-tor"] else 1  # Determine label
        labels.append(backward_label)  # Collect label

        # Add edges between forward and backward nodes
        G.add_edge(forward_node_id, backward_node_id)
        if previous_backward_node is not None:
            G.add_edge(previous_backward_node, forward_node_id)

        previous_backward_node = backward_node_id
        index += 1

    # Convert NetworkX graph to PyTorch Geometric data
    data = from_networkx(G)  
    # Extract node features
    node_features = [list(G.nodes[node_id].values()) for node_id in G.nodes]  
    # Convert features to tensor
    data.x = torch.tensor(node_features, dtype=torch.float)  
    # Convert labels to tensor
    data.y = torch.tensor(labels, dtype=torch.long) 
    # Collect graph data 
    pyg_graphs.append(data)  

# Scale the node features
# Fit scaler to all features
scaler.fit(all_features)  
for data in pyg_graphs:
    # Scale features

    data.x = torch.tensor(scaler.transform(data.x.numpy()), dtype=torch.float)  
# Define the training (cross-validation) function
def cross_validation_with_graphsage(pyg_graphs, k):
    # Initialize K-Fold cross-validation
    kf = KFold(n_splits=k, shuffle=True, random_state=42)  
    all_f1_scores = []
    all_auc_scores = []

    for fold, (train_index, test_index) in enumerate(kf.split(pyg_graphs)):
        print(f"Fold {fold + 1}/{k}")
        # Prepare training data
        train_dataset = [pyg_graphs[i] for i in train_index]  
        # Prepare test data
        test_dataset = [pyg_graphs[i] for i in test_index]  

        # Initialize training DataLoader
        train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True)  
         # Initialize test DataLoader

        test_loader = DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=False) 
        # Initialize model
        model = GraphSAGENodeModel(input_dim=pyg_graphs[0].x.shape[1], hidden_dim=FLAGS.dim_1, output_dim=2, num_layers=3)  
        # Move model to device (GPU or CPU)
        model = model.to(device)  
         # Initialize optimizer
        optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay)
        # Define loss function 
        criterion = nn.CrossEntropyLoss()  

        # Initialize learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)  
         # Define early stopping patience
        early_stopping_patience = 10 
        # Initialize best validation loss
        best_val_loss = float('inf')  
        # Initialize early stopping counter
        early_stopping_counter = 0  

        # Training loop
        for epoch in range(FLAGS.epochs):
            model.train()
            total_loss = 0
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out, node_embeddings = model(data)
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"  Epoch {epoch + 1}/{FLAGS.epochs}, Loss: {total_loss / len(train_loader):.4f}")

            # Validation
            model.eval()
            val_loss = 0
            val_preds = []
            val_labels = []
            val_probs = []
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    out, node_embeddings = model(data)
                    loss = criterion(out, data.y)
                    val_loss += loss.item()
                    probs = F.softmax(out, dim=1)[:, 1]
                    preds = out.argmax(dim=1)
                    val_probs.extend(probs.cpu().numpy())
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(data.y.cpu().numpy())
            # Step the learning rate scheduler
            scheduler.step(val_loss / len(test_loader))  
            f1 = f1_score(val_labels, val_preds, average="macro")
            auc = roc_auc_score(val_labels, val_probs)
            print(f"  Fold {fold + 1} Epoch {epoch + 1}, Val Loss: {val_loss / len(test_loader):.4f}, Val F1: {f1:.4f}, Val AUC: {auc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        all_f1_scores.append(f1)
        all_auc_scores.append(auc)

    avg_f1_score = sum(all_f1_scores) / len(all_f1_scores)
    avg_auc_score = sum(all_auc_scores) / len(all_auc_scores)
    return avg_f1_score, avg_auc_score

# Perform cross-validation(Testing)
avg_f1_score, avg_auc_score = cross_validation_with_graphsage(pyg_graphs, k=10)
print(f"Average F1 Score: {avg_f1_score:.4f}")
print(f"Average AUC Score: {avg_auc_score:.4f}")
