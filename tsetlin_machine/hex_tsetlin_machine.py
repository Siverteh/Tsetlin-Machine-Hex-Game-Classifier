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

    def __init__(self, board_size: int, dataset_path: Path, nrows: int, epochs: int = 100, num_clauses: int = 20, T: int = 200, s: int = 1.0, depth: int = 2):
        """
        Initialize the HexTsetlinMachine with dataset and hyperparameters.

        Parameters:
        - board_size (int): Size of the Hex board.
        - dataset_path (Path): Path to the dataset CSV file.
        - nrows (int): Number of rows to load from the dataset.
        """
        self.board_size = board_size  # Set the board size (e.g., 7 for 7x7)
        self.epochs = epochs
        self.num_clauses = num_clauses
        self.T = T
        self.s = s
        self.depth = depth

        # Initialize and load data using HexDataLoader
        self.data_loader = HexDataLoader(dataset_path, board_size)
        self.data_loader.load_data(desired_samples_per_class=nrows//2)  # Load specified number of rows

        # Retrieve feature matrix X and labels Y from the data loader
        self.X, self.Y = self.data_loader.get_all_data()
        self.X = self.X.reshape(-1, self.board_size, self.board_size)  # Reshape to (num_samples, board_size, board_size)
        self.Y = self.Y.astype(np.int32)  # Ensure labels are integers

        # Split data into training and testing sets (80% train, 20% test)
         # Split data into training (70%), validation (15%), and test (15%) sets
        X_temp, self.X_test, Y_temp, self.Y_test = train_test_split(
            self.X, self.Y, test_size=0.15, random_state=42, stratify=self.Y
        )

        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(
            X_temp, Y_temp, test_size=0.1765, random_state=42, stratify=Y_temp
        )

        # Map labels: -1 (Player 2) to 0, 1 (Player 1) to 1 for binary classification
        self.Y_train = np.where(self.Y_train == -1, 0, 1)
        self.Y_val = np.where(self.Y_val == -1, 0, 1)
        self.Y_test = np.where(self.Y_test == -1, 0, 1)

        # Set hyperparameters and configurations
        self.args = self._set_default_args()

        # Calculate the total number of nodes (cells) on the board
        self.number_of_nodes = self.board_size * self.board_size

        # Generate symbol names based on cell positions and their possible values
        self.symbol_names = self._set_symbol_names(property="multiple")

        # Initialize the MultiClassGraphTsetlinMachine with the specified hyperparameters
        self.tm = MultiClassGraphTsetlinMachine(
            self.args.number_of_clauses,
            self.args.T,
            self.args.s,
            depth=self.args.depth,
            message_size=self.args.message_size,
            message_bits=self.args.message_bits,
            max_included_literals=self.args.max_included_literals,
            grid=(16*13,1,1),
			block=(128,1,1)
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
        parser.add_argument("--epochs", default=self.epochs, type=int, help="Number of training epochs.")
        parser.add_argument("--number-of-clauses", default=self.num_clauses, type=int, help="Number of clauses in the Tsetlin Machine.")
        parser.add_argument("--T", default=self.T, type=int, help="Threshold for clause activation.")
        parser.add_argument("--s", default=self.s, type=float, help="Specificity parameter.")
        parser.add_argument("--depth", default=self.depth, type=int, help="Depth of the Tsetlin Machine.")
        parser.add_argument("--hypervector-size", default=256, type=int, help="Size of the hypervectors.")
        parser.add_argument("--hypervector-bits", default=2, type=int, help="Number of bits for hypervectors.")
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

    def _set_symbol_names(self, property: str = "single"):
        """
        Generate symbol names for each cell position and its possible values.

        Returns:
        - List[str]: List of symbol names.
        """
        symbol_names = []
        if property == "single":
            for i in range(self.board_size):
                for j in range(self.board_size):
                    for val in ["-1", "0", "1"]:
                        symbol_names.append(f"cell{i}_{j}_is_{val}")
        else:
            for i in range(self.board_size):
                for j in range(self.board_size):
                    symbol_names.append(f"cell{i}_{j}")
            for val in ["-1", "0", "1"]:
                symbol_names.append(val)
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
    
    def _get_edge_type(self, q, r, nq, nr):
        """
        Determine the edge type based on the direction from (q, r) to (nq, nr) in a hexagonal grid.
        
        Directions are:
        - 1: East (E)
        - 2: West (W)
        - 3: Northeast (NE)
        - 4: Southwest (SW)
        - 5: Southeast (SE)
        - 6: Northwest (NW)
        
        Parameters:
        - q, r: Coordinates of the current cell.
        - nq, nr: Coordinates of the neighbor cell.
        
        Returns:
        - int: Edge type corresponding to the direction.
        """
        if nq == q + 1 and nr == r:  # East (E)
            return 0
        elif nq == q - 1 and nr == r:  # West (W)
            return 1
        elif nq == q and nr == r + 1:  # Northeast (NE)
            return 2
        elif nq == q and nr == r - 1:  # Southwest (SW)
            return 3
        elif nq == q + 1 and nr == r - 1:  # Southeast (SE)
            return 4
        elif nq == q - 1 and nr == r + 1:  # Northwest (NW)
            return 5
        return -1  # Default case (if none match)

    def prepare_graphs(self):
        """
        Prepare graph structures for training, validation, and test data.
        This includes setting up nodes, edges, and node features for each graph.
        """
        # --- Prepare Training Graphs ---
        print("Setting up training graphs")
        self.graphs_train = Graphs(
            self.X_train.shape[0],                 # Number of graphs (samples) in training data
            symbols=self.symbol_names,               # List of feature names
            hypervector_size=self.args.hypervector_size,  # Size of hypervectors
            hypervector_bits=self.args.hypervector_bits   # Number of bits for hypervectors
        )
    
        # Set the number of nodes for each graph in training data
        for graph_id in range(self.X_train.shape[0]):
            self.graphs_train.set_number_of_graph_nodes(graph_id, self.number_of_nodes)
    
        # Prepare node configuration (internal setup required by Graphs)
        self.graphs_train.prepare_node_configuration()

        print("Adding training graph nodes")
        # Add nodes and edges to each graph in training data
        for graph_id in range(self.X_train.shape[0]):
            for q in range(self.board_size):
                for r in range(self.board_size):
                    node_id = f"cell{q}_{r}"
                    neighbors = self._get_neighbors(q, r)
                    # AddEach node has a number of edges equal to the number of neighbors
                    self.graphs_train.add_graph_node(graph_id, node_id, len(neighbors))
        
        # Prepare edge configuration (internal setup required by Graphs)
        self.graphs_train.prepare_edge_configuration()

        print("Adding training graph edges")
        # Add edges between nodes in training data graphs
        for graph_id in range(self.X_train.shape[0]):
            for q in range(self.board_size):
                for r in range(self.board_size):
                    node_id = f"cell{q}_{r}"  # Current cell ID
                    neighbors = self._get_neighbors(q, r)  # Retrieve valid neighbors for the current cell
                    for nq, nr in neighbors:
                        neighbor_id = f"cell{nq}_{nr}"  # Neighbor cell ID
                        edge_type = self._get_edge_type(q, r, nq, nr)
                        # Connect the current node to its neighbor with the appropriate edge type
                        self.graphs_train.add_graph_node_edge(graph_id, node_id, neighbor_id, edge_type)

        print("Adding training graph properties")
        # Add node features based on cell values in training data
        for graph_id in range(self.X_train.shape[0]):
            for q in range(self.board_size):
                for r in range(self.board_size):
                    node_id = f"cell{q}_{r}"  # Current cell ID
                    cell_value = str(self.X_train[graph_id][q][r])  # Value of the current cell as string
                    # AddAdd node properties: node_id (cell coordinate) and cell_value
                    self.graphs_train.add_graph_node_property(graph_id, node_id, node_id)
                    self.graphs_train.add_graph_node_property(graph_id, node_id, cell_value)

        print("Encoding training graph")
        # Encode the training graphs (finalize the graph structures)
        self.graphs_train.encode()

        print("Setting up validation graphs")
        # --- Prepare Validation Graphs ---
        self.graphs_val = Graphs(
            self.X_val.shape[0],
            init_with=self.graphs_train  # Inherit configurations from training graphs
        )
    
        # Set the number of nodes for each graph in validation data
        for graph_id in range(self.X_val.shape[0]):
            self.graphs_val.set_number_of_graph_nodes(graph_id, self.number_of_nodes)
    
        # Prepare node configuration for validation graphs
        self.graphs_val.prepare_node_configuration()

        print("Adding validation graph nodes")
        for graph_id in range(self.X_val.shape[0]):
            for q in range(self.board_size):
                for r in range(self.board_size):
                    node_id = f"cell{q}_{r}"
                    neighbors = self._get_neighbors(q, r)
                    self.graphs_val.add_graph_node(graph_id, node_id, len(neighbors))
    
        self.graphs_val.prepare_edge_configuration()

        print("Adding validation graph edges")
        for graph_id in range(self.X_val.shape[0]):
            for q in range(self.board_size):
                for r in range(self.board_size):
                    node_id = f"cell{q}_{r}"
                    for nq, nr in self._get_neighbors(q, r):
                        neighbor_id = f"cell{nq}_{nr}"
                        edge_type = self._get_edge_type(q, r, nq, nr)
                        self.graphs_val.add_graph_node_edge(graph_id, node_id, neighbor_id, edge_type)

        print("Adding validation graph properties")
        # AddAdd node properties
        for graph_id in range(self.X_val.shape[0]):
            for q in range(self.board_size):
                for r in range(self.board_size):
                    node_id = f"cell{q}_{r}"
                    cell_value = str(self.X_val[graph_id][q][r])
                    self.graphs_val.add_graph_node_property(graph_id, node_id, node_id)
                    self.graphs_val.add_graph_node_property(graph_id, node_id, cell_value)

        print("Encoding validation graph")
        self.graphs_val.encode()

        print("Setting up test graphs")
        # --- Prepare Test Graphs ---
        self.graphs_test = Graphs(
            self.X_test.shape[0],                 
            init_with=self.graphs_train          
        )
    
        for graph_id in range(self.X_test.shape[0]):
            self.graphs_test.set_number_of_graph_nodes(graph_id, self.number_of_nodes)
    
        # Prepare node configuration for test graphs
        self.graphs_test.prepare_node_configuration()

        print("Adding test graph nodes")
        for graph_id in range(self.X_test.shape[0]):
            for q in range(self.board_size):
                for r in range(self.board_size):
                    node_id = f"cell{q}_{r}"
                    neighbors = self._get_neighbors(q, r)
                    self.graphs_test.add_graph_node(graph_id, node_id, len(neighbors))
    
        self.graphs_test.prepare_edge_configuration()

        print("Adding test graph edges")
        for graph_id in range(self.X_test.shape[0]):
            for q in range(self.board_size):
                for r in range(self.board_size):
                    node_id = f"cell{q}_{r}"
                    for nq, nr in self._get_neighbors(q, r):
                        neighbor_id = f"cell{nq}_{nr}"
                        edge_type = self._get_edge_type(q, r, nq, nr)
                        self.graphs_test.add_graph_node_edge(graph_id, node_id, neighbor_id, edge_type)

        print("Adding test graph properties")
        # Add node properties
        for graph_id in range(self.X_test.shape[0]):
            for q in range(self.board_size):
                for r in range(self.board_size):
                    node_id = f"cell{q}_{r}"
                    cell_value = str(self.X_test[graph_id][q][r])
                    self.graphs_test.add_graph_node_property(graph_id, node_id, node_id)
                    self.graphs_test.add_graph_node_property(graph_id, node_id, cell_value)

        print("Encoding test graph")
        #Encode test graphs
        self.graphs_test.encode()
        print("Training, validation, and test data prepared")
    

    def train(self):
        """
        Train the Graph Tsetlin Machine over the specified number of epochs.
        After each epoch, evaluate and print training and validation accuracies along with timing.
        """
        for i in range(self.args.epochs):
            start_training = time()  # Start timer for training
    
            # Fit the model on training graphs for one epoch
            self.tm.fit(self.graphs_train, self.Y_train, epochs=1, incremental=True)
            stop_training = time()  # Stop timer after training
    
            #Evaluate on validation set
            start_validation = time()
            result_val = 100 * (self.tm.predict(self.graphs_val) == self.Y_val).mean()
            stop_validation = time()
    
            # Predict on training graphs and calculate training accuracy
            result_train = 100 * (self.tm.predict(self.graphs_train) == self.Y_train).mean()
    
            # Print epoch number, training accuracy, validation accuracy, and timings
            print(f"Epoch {i}: Train Acc: {result_train:.2f}%, Val Acc: {result_val:.2f}%, Training Time: {stop_training - start_training:.2f}s, Validation Time: {stop_validation - start_validation:.2f}s")

        # After all epochs, retrieve and print clause information
        #weights = self.tm.get_state()[1].reshape(2, -1)  # Retrieve weights for clauses, reshaped for two classes
        #for i in range(self.tm.number_of_clauses):
        #    # Print clause number and its weights for both classes
        #    print(f"Clause #{i} W:({weights[0, i]} {weights[1, i]})", end=' ')
        #    literals = []  # List to hold literals (features) in the clause
        #    # Iterate over each possible literal in the clause
        #    for k in range(self.args.hypervector_size * 2):
        #        if self.tm.ta_action(0, i, k):  # Check if the literal is included in the clause
        #            if k < self.args.hypervector_size:
        #                # Positive literal
        #                symbol_name = self.symbol_names[k]
        #                literals.append(f"{symbol_name}")
        #            else:
        #                # Negative literal
        #                symbol_name = self.symbol_names[k - self.args.hypervector_size]
        #                literals.append(f"NOT {symbol_name}")
        #    # Join literals with ' AND ' to represent the clause
        #   print(" AND ".join(literals))


    def evaluate(self):
        """
        Perform a final evaluation after training and print the final accuracies.
        """
        # Start timer for final testing
        start_testing = time()
        # Predict on test graphs and calculate test accuracy
        result_test = 100 * (self.tm.predict(self.graphs_val) == self.Y_val).mean()
        stop_testing = time()
    
        # Predict on training graphs and calculate training accuracy
        result_train = 100 *(self.tm.predict(self.graphs_train) == self.Y_train).mean()
    
        # Print final training and test accuracies along with timing
        print(f"Final Results: Train Accuracy = {result_train:.2f}%, Test Accuracy = {result_test:.2f}%, Testing Time = {stop_testing - start_testing:.2}s")
    

    def test(self):
        """
        Print the predicted and actual winners for each game in the test set.
        """
        # Predict the outcomes on the test set
        predictions = self.tm.predict(self.graphs_test)

        correct = 0
        count = 0
    
        # Print predicted and actual winners for each test sample
        for idx in range(len(predictions)):
            predicted_winner = predictions[idx]
            actual_winner = self.Y_test[idx]
            # Map labels back to Player 1 and Player 2
            predicted_player = 1 if predicted_winner == 1 else 2
            actual_player = 1 if actual_winner == 1 else 2

            if predicted_player == actual_player:
                correct += 1
            count += 1
            if (idx % 500) == 0:
                print(f"Game {idx + 1}: Predicted Winner: Player {predicted_player}, Actual Winner: Player {actual_player}")
                
        correct_count_ratio = correct / count
        print(f"Correct: {correct}/{count}, Percentage: {correct_count_ratio*100}%")


if __name__ == "__main__":
    """
    Main execution block to instantiate the HexTsetlinMachine, prepare graphs, train the model,
    and evaluate its performance.
    """
    # 5X5 100% Accuracy
    """hexTsetlinMachine = HexTsetlinMachine(
        board_size=7,  # Size of the Hex board (7x7)
        dataset_path="hex_games_1_000_000_size_7.csv",  # Path to the dataset CSV file
        nrows=100000,  # Number of rows to load from the dataset,
        epochs=30,
        num_clauses=600,
        T=300,
        s=1.0,
        depth=1
    )"""

    # 7X7 100% Accuracy
    """hexTsetlinMachine = HexTsetlinMachine(
        board_size=5,  # Size of the Hex board (7x7)
        dataset_path="hex_winning_positions.csv",  # Path to the dataset CSV file
        nrows=80000,  # Number of rows to load from the dataset,
        epochs=10,
        num_clauses=1400,
        T=700,
        s=1.0,
        depth=1
    )"""

    #9X9 100% accuracy
    hexTsetlinMachine = HexTsetlinMachine(
        board_size=9,  # Size of the Hex board (7x7)
        dataset_path="hex_games_9.csv",  # Path to the dataset CSV file
        nrows=400000,  # Number of rows to load from the dataset,
        epochs=30,
        num_clauses=20000,
        T=5000,
        s=1.0,
        depth=1
    )
    
    #11X11
    """hexTsetlinMachine = HexTsetlinMachine(
        board_size=11,  # Size of the Hex board (7x7)
        dataset_path="hex_games_11.csv",  # Path to the dataset CSV file
        nrows=100000,  # Number of rows to load from the dataset,
        epochs=30,
        num_clauses=2000,
        T=500,
        s=1.0,
        depth=1
    )"""

    #13X13
    """hexTsetlinMachine = HexTsetlinMachine(
        board_size=13,  # Size of the Hex board (7x7)
        dataset_path="hex_games_13.csv",  # Path to the dataset CSV file
        nrows=100000,  # Number of rows to load from the dataset,
        epochs=30,
        num_clauses=2500,
        T=1250,
        s=1.0,
        depth=1
    )"""

    # Prepare the graph structures for training and testing data
    hexTsetlinMachine.prepare_graphs()

    # Train the GTM model
    hexTsetlinMachine.train()

    # Evaluate the model's performance after training
    hexTsetlinMachine.evaluate()

    hexTsetlinMachine.test()
