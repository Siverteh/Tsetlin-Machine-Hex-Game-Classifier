from sklearn.model_selection import train_test_split
import pandas as pd
import random
from GraphTsetlinMachine.graphs import Graphs
import numpy as np
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from time import time
from hex_dataset_loader import HexDataLoader
import argparse

# Neighbor offsets for a hex grid in axial coordinates (q, r)
hex_neighbors = [
    (+1,  0), (-1,  0), 
    (0,  +1), (0,  -1), 
    (+1, -1), (-1, +1)
]

def get_neighbors(q, r, board_size):
    """Get the neighbors for a hex cell in axial coordinates."""
    neighbors = []
    for dq, dr in hex_neighbors:
        nq, nr = q + dq, r + dr
        if 0 <= nq < board_size and 0 <= nr < board_size:
            neighbors.append((nq, nr))
    return neighbors

# Load Hex dataset
data_loader = HexDataLoader('datasets/hex_games_1_000_000_size_7.csv', board_size=7)
data_loader.load_data(nrows=1000)  # Use a smaller dataset for testing
X, Y = data_loader.get_all_data()

# Convert data to appropriate formats
X = X.reshape(-1, 7, 7)
Y = Y.astype(np.int32)

# Split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Map labels: -1 -> 0 (player 2), 1 -> 1 (player 1)
Y_train = np.where(Y_train == -1, 0, 1)
Y_test = np.where(Y_test == -1, 0, 1)

print(f"Train distribution: {np.bincount(Y_train)}")
print(f"Test distribution: {np.bincount(Y_test)}")

# Set default arguments
def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--number-of-clauses", default=1000, type=int)
    parser.add_argument("--T", default=5000, type=int)
    parser.add_argument("--s", default=5.0, type=float)
    parser.add_argument("--depth", default=1, type=int)
    parser.add_argument("--hypervector-size", default=None, type=int)
    parser.add_argument("--hypervector-bits", default=2, type=int)
    parser.add_argument("--message-size", default=128, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument("--max-included-literals", default=16, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

args = default_args()

# Update for Hex board size
board_size = 7
number_of_nodes = board_size * board_size

# Use cell names as symbols (without values)
symbol_names = [f"cell{i}_{j}" for i in range(board_size) for j in range(board_size)]
args.hypervector_size = len(symbol_names)

# Prepare graphs for training data
graphs_train = Graphs(
    X_train.shape[0],
    symbol_names=symbol_names,
    hypervector_size=args.hypervector_size,
    hypervector_bits=args.hypervector_bits,
)

for graph_id in range(X_train.shape[0]):
    graphs_train.set_number_of_graph_nodes(graph_id, number_of_nodes)

graphs_train.prepare_node_configuration()

# Add nodes to the graphs
for graph_id in range(X_train.shape[0]):
    for q in range(board_size):
        for r in range(board_size):
            node_id = f"cell{q}_{r}"
            graphs_train.add_graph_node(graph_id, node_id, 6)

# Add edges between nodes
graphs_train.prepare_edge_configuration()

for graph_id in range(X_train.shape[0]):
    for q in range(board_size):
        for r in range(board_size):
            node_id = f"cell{q}_{r}"
            neighbors = get_neighbors(q, r, board_size)
            for nq, nr in neighbors:
                neighbor_id = f"cell{nq}_{nr}"
                graphs_train.add_graph_node_edge(
                    graph_id, node_id, neighbor_id, 1
                )

# Add features to nodes, mapping cell values to symbols
for graph_id in range(X_train.shape[0]):
    for q in range(board_size):
        for r in range(board_size):
            node_id = f"cell{q}_{r}"
            cell_value = int(X_train[graph_id][q][r])
            # Map cell value to a binary feature
            if cell_value == 1:
                graphs_train.add_graph_node_feature(graph_id, node_id, node_id)
            elif cell_value == -1:
                # Optionally, add a 'NOT' feature or skip
                pass
            # For empty cells (0), you may decide to add or skip features

graphs_train.encode()

# Prepare graphs for test data
graphs_test = Graphs(X_test.shape[0], init_with=graphs_train)

for graph_id in range(X_test.shape[0]):
    graphs_test.set_number_of_graph_nodes(graph_id, number_of_nodes)

graphs_test.prepare_node_configuration()

# Add nodes to the test graphs
for graph_id in range(X_test.shape[0]):
    for q in range(board_size):
        for r in range(board_size):
            node_id = f"cell{q}_{r}"
            graphs_test.add_graph_node(graph_id, node_id, 6)

# Add edges between nodes for test data
graphs_test.prepare_edge_configuration()

for graph_id in range(X_test.shape[0]):
    for q in range(board_size):
        for r in range(board_size):
            node_id = f"cell{q}_{r}"
            neighbors = get_neighbors(q, r, board_size)
            for nq, nr in neighbors:
                neighbor_id = f"cell{nq}_{nr}"
                graphs_test.add_graph_node_edge(
                    graph_id, node_id, neighbor_id, 1
                )

# Add features to nodes, mapping cell values to symbols
for graph_id in range(X_test.shape[0]):
    for q in range(board_size):
        for r in range(board_size):
            node_id = f"cell{q}_{r}"
            cell_value = int(X_test[graph_id][q][r])
            if cell_value == 1:
                graphs_test.add_graph_node_feature(graph_id, node_id, node_id)
            elif cell_value == -1:
                pass  # Handle as needed

graphs_test.encode()

print("Training and testing data prepared")

# Initialize the Tsetlin machine
tm = MultiClassGraphTsetlinMachine(
    args.number_of_clauses,
    args.T,
    args.s,
    depth=args.depth,
    message_size=args.message_size,
    message_bits=args.message_bits,
    max_included_literals=args.max_included_literals,
)

# Training and Testing
for i in range(args.epochs):
    start_training = time()
    tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    result_test = 100 * (tm.predict(graphs_test) == Y_test).mean()
    stop_testing = time()

    result_train = 100 * (tm.predict(graphs_train) == Y_train).mean()

    print(
        f"Epoch {i} - Train Accuracy: {result_train:.2f}%, "
        f"Test Accuracy: {result_test:.2f}%, "
        f"Training Time: {stop_training - start_training:.2f}s, "
        f"Testing Time: {stop_testing - start_testing:.2f}s"
    )