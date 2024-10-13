from sklearn.model_selection import train_test_split
import pandas as pd
import random
from GraphTsetlinMachine.graphs import Graphs
import numpy as np
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from time import time
from hex_dataset_loader import HexDataLoader
import argparse
from pathlib import Path

# Neighbor offsets for a hex grid in axial coordinates (q, r)
hex_neighbors = [
    (+1,  0), (-1,  0),  # East and West neighbors
    (0,  +1), (0,  -1),  # Northeast and Southwest neighbors
    (+1, -1), (-1, +1)   # Southeast and Northwest neighbors
]

class HexTsetlinMachine():
    """
    A class to encapsulate the Graph Tsetlin Machine (GTM) for predicting Hex game outcomes.

    Attributes:
    - board_size (int): Size of the Hex board (e.g., 7 for a 7x7 board).
    - data_loader (HexDataLoader): Instance to load and process the Hex dataset.
    - X (np.ndarray): Feature matrix containing game states.
    - Y (np.ndarray): Labels indicating the winner of each game.
    - X_train, X_test (np.ndarray): Training and testing feature matrices.
    - Y_train, Y_test (np.ndarray): Training and testing labels.
    - args (argparse.Namespace): Hyperparameters and configurations.
    - number_of_nodes (int): Total number of nodes (cells) in the graph.
    - symbol_names (List[str]): List of feature names for the GTM.
    - tm (MultiClassGraphTsetlinMachine): The GTM model instance.
    """

    def __init__(self, board_size: int, dataset_path: Path, nrows: int):
        """
        Initialize the HexTsetlinMachine with dataset and hyperparameters.

        Parameters:
        - board_size (int): Size of the Hex board.
        - dataset_path (Path): Path to the dataset CSV file.
        - nrows (int): Number of rows to load from the dataset.
        """
        self.board_size = board_size  # Set the board size (e.g., 7 for 7x7)

        # Initialize and load data using HexDataLoader
        self.data_loader = HexDataLoader(dataset_path, board_size)
        self.data_loader.load_data(nrows=nrows)  # Load specified number of rows

        # Retrieve feature matrix X and labels Y from the data loader
        self.X, self.Y = self.data_loader.get_all_data()
        self.X = self.X.reshape(-1, self.board_size, self.board_size)  # Reshape to (num_samples, board_size, board_size)
        self.Y = self.Y.astype(np.int32)  # Ensure labels are integers

        # Split data into training and testing sets (80% train, 20% test)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=0.2, random_state=42
        )

        # Map labels: -1 (Player 2) to 0, 1 (Player 1) to 1 for binary classification
        self.Y_train = np.where(self.Y_train == -1, 0, 1)
        self.Y_test = np.where(self.Y_test == -1, 0, 1)

        # Set hyperparameters and configurations
        self.args = self._set_default_args()

        # Calculate the total number of nodes (cells) on the board
        self.number_of_nodes = self.board_size * self.board_size

        # Generate symbol names based on cell positions and their possible values
        self.symbol_names = self._set_symbol_names()

        # Initialize the MultiClassGraphTsetlinMachine with the specified hyperparameters
        self.tm = MultiClassGraphTsetlinMachine(
            self.args.number_of_clauses,
            self.args.T,
            self.args.s,
            depth=self.args.depth,
            message_size=self.args.message_size,
            message_bits=self.args.message_bits,
            max_included_literals=self.args.max_included_literals
        )

    def _set_default_args(self, **kwargs):
        """
        Define and parse default hyperparameters for the Tsetlin Machine.

        Parameters:
        - kwargs: Additional keyword arguments to override defaults.

        Returns:
        - argparse.Namespace: Parsed arguments with hyperparameters.
        """
        parser = argparse.ArgumentParser(description="Hex Graph Tsetlin Machine Hyperparameters")

        # Define command-line arguments with default values and help descriptions
        parser.add_argument("--epochs", default=100, type=int, help="Number of training epochs.")
        parser.add_argument("--number-of-clauses", default=30000, type=int, help="Number of clauses in the Tsetlin Machine.")
        parser.add_argument("--T", default=1000, type=int, help="Threshold for clause activation.")
        parser.add_argument("--s", default=5.0, type=float, help="Specificity parameter.")
        parser.add_argument("--depth", default=5, type=int, help="Depth of the Tsetlin Machine.")
        parser.add_argument("--hypervector-size", default=self.board_size*self.board_size, type=int, help="Size of the hypervectors.")
        parser.add_argument("--hypervector-bits", default=8, type=int, help="Number of bits for hypervectors.")
        parser.add_argument("--message-size", default=256, type=int, help="Size of the messages.")
        parser.add_argument("--message-bits", default=2, type=int, help="Number of bits for messages.")
        parser.add_argument("--max-included-literals", default=32, type=int, help="Maximum number of literals included in clauses.")

        # Parse arguments without reading from the command line (empty list)
        args = parser.parse_args(args=[])

        # Override default arguments with any provided keyword arguments
        for key, value in kwargs.items():
            if key in args.__dict__:
                setattr(args, key, value)
        return args

    def _set_symbol_names(self):
        """
        Generate symbol names for each cell position and its possible values.

        Returns:
        - List[str]: List of symbol names.
        """
        symbol_names = []
        # Iterate over each cell position on the board
        for i in range(self.board_size):
            for j in range(self.board_size):
                for val in [-1, 0, 1]:  # Possible cell values: -1 (Player 2), 0 (Empty), 1 (Player 1)
                    # Create a unique symbol name for each cell-value combination
                    symbol_names.append(f"cell{i}_{j}_is_{val}")
        return symbol_names

    def _get_neighbors(self, q, r):
        """
        Get the neighbors for a hex cell in axial coordinates.

        Parameters:
        - q (int): Axial coordinate q of the cell.
        - r (int): Axial coordinate r of the cell.

        Returns:
        - List[Tuple[int, int]]: List of neighboring cell coordinates.
        """
        neighbors = []
        # Iterate through predefined neighbor offsets
        for dq, dr in hex_neighbors:
            nq, nr = q + dq, r + dr  # Calculate neighbor coordinates
            # Check if the neighbor is within the board boundaries
            if 0 <= nq < self.board_size and 0 <= nr < self.board_size:
                neighbors.append((nq, nr))  # Add valid neighbor to the list
        return neighbors

    def prepare_graphs(self):
        """
        Prepare graph structures for both training and testing data.
        This includes setting up nodes, edges, and node features for each graph.
        """
        # Initialize Graphs object for training data
        self.graphs_train = Graphs(
            self.X_train.shape[0],                 # Number of graphs (samples) in training data
            symbol_names=self.symbol_names,        # List of feature names
            hypervector_size=self.args.hypervector_size,  # Size of hypervectors
            hypervector_bits=self.args.hypervector_bits   # Number of bits for hypervectors
        )

        # Set the number of nodes for each graph in training data
        for graph_id in range(self.X_train.shape[0]):
            self.graphs_train.set_number_of_graph_nodes(graph_id, self.number_of_nodes)

        # Prepare node configuration (internal setup required by Graphs)
        self.graphs_train.prepare_node_configuration()

        # Add nodes to each graph in training data
        for graph_id in range(self.X_train.shape[0]):
            for q in range(self.board_size):
                for r in range(self.board_size):
                    node_id = f"cell{q}_{r}"
                    neighbors = self._get_neighbors(q, r)
                    self.graphs_train.add_graph_node(graph_id, node_id, len(neighbors))

        # Prepare edge configuration (internal setup required by Graphs)
        self.graphs_train.prepare_edge_configuration()

        # Add edges between nodes in training data graphs
        for graph_id in range(self.X_train.shape[0]):
            for q in range(self.board_size):
                for r in range(self.board_size):
                    node_id = f"cell{q}_{r}"  # Current cell ID
                    neighbors = self._get_neighbors(q, r)  # Retrieve valid neighbors for the current cell
                    for nq, nr in neighbors:
                        neighbor_id = f"cell{nq}_{nr}"  # Neighbor cell ID
                        # Connect the current node to its neighbor with edge type 1
                        self.graphs_train.add_graph_node_edge(graph_id, node_id, neighbor_id, 1)

        # Add node features based on cell values in training data
        for graph_id in range(self.X_train.shape[0]):
            for q in range(self.board_size):
                for r in range(self.board_size):
                    node_id = f"cell{q}_{r}"  # Current cell ID
                    cell_value = self.X_train[graph_id][q][r]  # Value of the current cell
                    # Create a feature name that indicates the cell's value
                    feature_name = f"cell{q}_{r}_is_{int(cell_value)}"
                    # Add the feature to the node in the graph
                    self.graphs_train.add_graph_node_feature(graph_id, node_id, feature_name)

        # Encode the training graphs (finalize the graph structures)
        self.graphs_train.encode()

        # Initialize Graphs object for testing data, inheriting configurations from training graphs
        self.graphs_test = Graphs(
            self.X_test.shape[0],                 
            init_with=self.graphs_train          
        )

        # Set the number of nodes for each graph in testing data
        for graph_id in range(self.X_test.shape[0]):
            self.graphs_test.set_number_of_graph_nodes(graph_id, self.number_of_nodes)

        # Prepare node configuration for testing graphs
        self.graphs_test.prepare_node_configuration()

        # Add nodes to each graph in testing data
        for graph_id in range(self.X_test.shape[0]):
            for q in range(self.board_size):
                for r in range(self.board_size):
                    node_id = f"cell{q}_{r}"
                    neighbors = self._get_neighbors(q, r)
                    # Each node (cell) has 6 neighbors in a hex grid
                    self.graphs_test.add_graph_node(graph_id, node_id, len(neighbors))

        # Prepare edge configuration for testing graphs
        self.graphs_test.prepare_edge_configuration()

        # Add edges between nodes in testing data graphs
        for graph_id in range(self.X_test.shape[0]):
            for q in range(self.board_size):
                for r in range(self.board_size):
                    node_id = f"cell{q}_{r}"  # Current cell ID
                    neighbors = self._get_neighbors(q, r)  # Retrieve valid neighbors for the current cell
                    for nq, nr in neighbors:
                        neighbor_id = f"cell{nq}_{nr}"  # Neighbor cell ID
                        # Connect the current node to its neighbor with edge type 1
                        self.graphs_test.add_graph_node_edge(graph_id, node_id, neighbor_id, 1)

        # Add node features based on cell values in testing data
        for graph_id in range(self.X_test.shape[0]):
            for q in range(self.board_size):
                for r in range(self.board_size):
                    node_id = f"cell{q}_{r}"  # Current cell ID
                    cell_value = self.X_test[graph_id][q][r]  # Value of the current cell
                    # Create a feature name that indicates the cell's value
                    feature_name = f"cell{q}_{r}_is_{int(cell_value)}"
                    # Add the feature to the node in the graph
                    self.graphs_test.add_graph_node_feature(graph_id, node_id, feature_name)

        # Encode the testing graphs (finalize the graph structures)
        self.graphs_test.encode()
        print("Training and testing data prepared")

    def train(self):
        """
        Train the Graph Tsetlin Machine over the specified number of epochs.
        After each epoch, evaluate and print training and testing accuracies along with timing.
        """
        # Iterate over the number of training epochs
        for i in range(self.args.epochs):
            start_training = time()  # Start timer for training

            # Fit the model on training graphs for one epoch
            self.tm.fit(self.graphs_train, self.Y_train, epochs=1, incremental=True)
            stop_training = time()  # Stop timer after training

            start_testing = time()  # Start timer for testing

            # Predict on test graphs and calculate test accuracy
            result_test = 100 * (self.tm.predict(self.graphs_test) == self.Y_test).mean()
            stop_testing = time()  # Stop timer after testing

            # Predict on training graphs and calculate training accuracy
            result_train = 100 * (self.tm.predict(self.graphs_train) == self.Y_train).mean()

            # Print epoch number, training accuracy, test accuracy, and timings
            print(f"{i} {result_train:.2f} {result_test:.2f} {stop_training - start_training:.2f} {stop_testing - start_testing:.2f}")

        # After all epochs, retrieve and print clause information
        weights = self.tm.get_state()[1].reshape(2, -1)  # Retrieve weights for clauses, reshaped for two classes
        for i in range(self.tm.number_of_clauses):
            # Print clause number and its weights for both classes
            print(f"Clause #{i} W:({weights[0, i]} {weights[1, i]})", end=' ')
            literals = []  # List to hold literals (features) in the clause
            # Iterate over each possible literal in the clause
            for k in range(self.args.hypervector_size * 2):
                if self.tm.ta_action(0, i, k):  # Check if the literal is included in the clause
                    if k < self.args.hypervector_size:
                        # Positive literal: cell is occupied by Player 1
                        literals.append(f"cell{k // self.board_size}_{k % self.board_size}")
                    else:
                        # Negative literal: cell is not occupied by Player 1 (Player 2 or Empty)
                        literals.append(f"NOT cell{(k - self.args.hypervector_size) // self.board_size}_{(k - self.args.hypervector_size) % self.board_size}")
            # Join literals with ' AND ' to represent the clause
            print(" AND ".join(literals))

    def evaluate(self):
        """
        Perform a final evaluation after training and print the final accuracies.
        Also, optionally print the hypervectors for further analysis.
        """
        # Start timer for final training (one additional epoch)
        start_training = time()
        self.tm.fit(self.graphs_train, self.Y_train, epochs=1, incremental=True)
        stop_training = time()

        # Start timer for final testing
        start_testing = time()
        # Predict on test graphs and calculate test accuracy
        result_test = 100 * (self.tm.predict(self.graphs_test) == self.Y_test).mean()
        stop_testing = time()

        # Predict on training graphs and calculate training accuracy
        result_train = 100 * (self.tm.predict(self.graphs_train) == self.Y_train).mean()

        # Print final training and testing accuracies along with timings
        print(f"Final Results: Train Accuracy = {result_train:.2f}%, Test Accuracy = {result_test:.2f}%, Time = {stop_training - start_training:.2f}s (training), {stop_testing - start_testing:.2f}s (testing)")

        # Optionally, print hypervectors (useful for debugging or analysis)
        print(self.graphs_train.hypervectors)

if __name__ == "__main__":
    """
    Main execution block to instantiate the HexTsetlinMachine, prepare graphs, train the model,
    and evaluate its performance.
    """
    # Instantiate the HexTsetlinMachine with specified parameters
    hexTsetlinMachine = HexTsetlinMachine(
        board_size=7,  # Size of the Hex board (7x7)
        dataset_path=Path("datasets/hex_games_1_000_000_size_7.csv"),  # Path to the dataset CSV file
        nrows=1000  # Number of rows to load from the dataset
    )

    # Prepare the graph structures for training and testing data
    hexTsetlinMachine.prepare_graphs()

    # Train the GTM model
    hexTsetlinMachine.train()

    # Evaluate the model's performance after training
    hexTsetlinMachine.evaluate()
