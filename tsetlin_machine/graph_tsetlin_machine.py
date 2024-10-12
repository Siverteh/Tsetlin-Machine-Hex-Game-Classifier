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
data_loader = HexDataLoader('datasets/hex_games_1_000_000_size_7.csv', board_size=7)  # Adjusting board size to 7
data_loader.load_data(nrows=50000)
X, Y = data_loader.get_all_data()

# Reshape data if necessary
X = X.reshape(-1, 7, 7)
Y = Y.astype(np.int32)

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Map labels: -1 -> 0 (player 2), 1 -> 1 (player 1)
Y_train = np.where(Y_train == -1, 0, 1)
Y_test = np.where(Y_test == -1, 0, 1)

print(f"Train distribution: {np.bincount(Y_train)}")
print(f"Test distribution: {np.bincount(Y_test)}")


# Set default arguments
def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--number-of-clauses", default=20000, type=int)
    parser.add_argument("--T", default=25000, type=int)
    parser.add_argument("--s", default=10.0, type=float)
    parser.add_argument("--depth", default=1, type=int)
    parser.add_argument("--hypervector-size", default=128, type=int)
    parser.add_argument("--hypervector-bits", default=2, type=int)
    parser.add_argument("--message-size", default=256, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument("--max-included-literals", default=32, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

args = default_args()

# Update for Hex board size
board_size = 7

number_of_nodes = board_size * board_size  # Each hexagon is a node

# Use cell names from the dataset instead of generic symbols
symbol_names = []
for i in range(board_size):
    for j in range(board_size):
        for val in [-1, 0, 1]:
            symbol_names.append(f"cell{i}_{j}_is_{val}")

# Prepare graphs for training data
graphs_train = Graphs(X_train.shape[0], symbol_names=symbol_names, hypervector_size=args.hypervector_size, hypervector_bits=args.hypervector_bits)

for graph_id in range(X_train.shape[0]):
    graphs_train.set_number_of_graph_nodes(graph_id, number_of_nodes)

graphs_train.prepare_node_configuration()

# First, add all nodes to the graph
for graph_id in range(X_train.shape[0]):
    for q in range(board_size):
        for r in range(board_size):
            node_id = f"cell{q}_{r}"
            graphs_train.add_graph_node(graph_id, node_id, 6)  # 6 because hexes have 6 neighbors

# Now, add the edges between nodes
graphs_train.prepare_edge_configuration()

for graph_id in range(X_train.shape[0]):
    for q in range(board_size):
        for r in range(board_size):
            node_id = f"cell{q}_{r}"
            # Connect each node to its hex neighbors
            neighbors = get_neighbors(q, r, board_size)
            for nq, nr in neighbors:
                neighbor_id = f"cell{nq}_{nr}"
                graphs_train.add_graph_node_edge(graph_id, node_id, neighbor_id, 1)  # edge_type can be 1 or some other value

for graph_id in range(X_train.shape[0]):
    for q in range(board_size):
        for r in range(board_size):
            node_id = f"cell{q}_{r}"
            cell_value = X_train[graph_id][q][r]
            feature_name = f"cell{q}_{r}_is_{int(cell_value)}"
            graphs_train.add_graph_node_feature(graph_id, node_id, feature_name)

graphs_train.encode()

# Prepare graphs for test data
graphs_test = Graphs(X_test.shape[0], init_with=graphs_train)
for graph_id in range(X_test.shape[0]):
    graphs_test.set_number_of_graph_nodes(graph_id, number_of_nodes)

graphs_test.prepare_node_configuration()

# First, add all nodes to the test graph
for graph_id in range(X_test.shape[0]):
    for q in range(board_size):
        for r in range(board_size):
            node_id = f"cell{q}_{r}"
            graphs_test.add_graph_node(graph_id, node_id, 6)  # 6 because hexes have 6 neighbors

# Now, add the edges between nodes for test data
graphs_test.prepare_edge_configuration()

for graph_id in range(X_test.shape[0]):
    for q in range(board_size):
        for r in range(board_size):
            node_id = f"cell{q}_{r}"
            # Connect each node to its hex neighbors
            neighbors = get_neighbors(q, r, board_size)
            for nq, nr in neighbors:
                neighbor_id = f"cell{nq}_{nr}"
                graphs_test.add_graph_node_edge(graph_id, node_id, neighbor_id, 1)  # edge_type can be 1 or some other value

for graph_id in range(X_test.shape[0]):
    for q in range(board_size):
        for r in range(board_size):
            node_id = f"cell{q}_{r}"
            cell_value = X_test[graph_id][q][r]
            feature_name = f"cell{q}_{r}_is_{int(cell_value)}"
            graphs_test.add_graph_node_feature(graph_id, node_id, feature_name)

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
    max_included_literals=args.max_included_literals
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

    print(f"{i} {result_train:.2f} {result_test:.2f} {stop_training - start_training:.2f} {stop_testing - start_testing:.2f}")

# Output clause information
weights = tm.get_state()[1].reshape(2, -1)
for i in range(tm.number_of_clauses):
    print(f"Clause #{i} W:({weights[0, i]} {weights[1, i]})", end=' ')
    l = []
    for k in range(args.hypervector_size * 2):
        if tm.ta_action(0, i, k):
            if k < args.hypervector_size:
                l.append(f"cell{k // board_size}_{k % board_size}")
            else:
                l.append(f"NOT cell{(k - args.hypervector_size) // board_size}_{(k - args.hypervector_size) % board_size}")
    print(" AND ".join(l))

# Final training and testing run after clause information
start_training = time()
tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
stop_training = time()

start_testing = time()
result_test = 100 * (tm.predict(graphs_test) == Y_test).mean()
stop_testing = time()

result_train = 100 * (tm.predict(graphs_train) == Y_train).mean()

print(f"Final Results: Train Accuracy = {result_train:.2f}%, Test Accuracy = {result_test:.2f}%, Time = {stop_training - start_training:.2f} (training), {stop_testing - start_testing:.2f} (testing)")

print(graphs_train.hypervectors)

# Load the test data from CSV
test_data = pd.read_csv('test.csv')

# Extract features (board state) and ignore the 'winner' column
X_test_new = test_data.drop('winner', axis=1).values
X_test_new = X_test_new.reshape(-1, board_size, board_size)  # Reshape to the 7x7 board

# Prepare a graph for each test game and predict the winner
for idx, game_state in enumerate(X_test_new):
    # Prepare the current game state as a graph
    graphs_test = Graphs(1, symbol_names=symbol_names, hypervector_size=args.hypervector_size, hypervector_bits=args.hypervector_bits)
    graphs_test.set_number_of_graph_nodes(0, number_of_nodes)
    graphs_test.prepare_node_configuration()

    # Add the nodes for the current game state
    for q in range(board_size):
        for r in range(board_size):
            node_id = f"cell{q}_{r}"
            graphs_test.add_graph_node(0, node_id, 6)  # 6 because hexes have 6 neighbors

    # Add the edges between nodes for the current game state
    graphs_test.prepare_edge_configuration()
    for q in range(board_size):
        for r in range(board_size):
            node_id = f"cell{q}_{r}"
            neighbors = get_neighbors(q, r, board_size)
            for nq, nr in neighbors:
                neighbor_id = f"cell{nq}_{nr}"
                graphs_test.add_graph_node_edge(0, node_id, neighbor_id, 1)

    # Add the features for the current game state
    for q in range(board_size):
        for r in range(board_size):
            node_id = f"cell{q}_{r}"
            cell_value = game_state[q][r]
            feature_name = f"cell{q}_{r}_is_{int(cell_value)}"
            graphs_test.add_graph_node_feature(0, node_id, feature_name)

    graphs_test.encode()

    # Predict the outcome of the current game state
    predicted_winner = tm.predict(graphs_test)[0]  # Get the prediction for this single graph

    # Print the predicted winner
    if predicted_winner == 1:
        print(f"Game {idx + 1}: Predicted Winner: Player 1")
    else:
        print(f"Game {idx + 1}: Predicted Winner: Player 2")
