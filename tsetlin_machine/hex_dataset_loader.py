import pandas as pd
import random

class HexDataLoader:
    """
    A class to load and handle the Hex game dataset.
    """
    def __init__(self, data_file, board_size=7):
        """
        Initializes the data loader with the dataset file path and board size.
        
        Parameters:
        - data_file: Path to the CSV file containing the dataset.
        - board_size: Size of the Hex board (7 for 7x7 or 11 for 11x11).
        """
        self.data_file = data_file
        self.board_size = board_size
        self.data = None
        self.X = None
        self.y = None

    def load_data(self, nrows=1000):
        """
        Loads the dataset from the CSV file.
        """
        # Read the CSV file into a pandas DataFrame
        self.data = pd.read_csv(self.data_file, nrows=nrows)

        # Separate features (board states) and labels (winner)
        self.X = self.data.drop('winner', axis=1)  # All columns except 'winner'
        self.y = self.data['winner']  # The 'winner' column

    def get_random_entry(self):
        """
        Retrieves a random game state and its winner.
        
        Returns:
        - board_state: A flat representation of the board state (as a NumPy array).
        - winner: The winner of the game (1 for player 1, -1 for player 2).
        """
        idx = random.randint(0, len(self.X) - 1)
        board_state = self.X.iloc[idx]  # Return as NumPy array for easy manipulation
        winner = self.y.iloc[idx]
        return board_state, winner

    def get_all_data(self):
        """
        Retrieves the entire dataset in a format ready for model input.
        
        Returns:
        - X: The board states as a NumPy array.
        - y: The game winners as a NumPy array.
        """
        X = self.X.values  # Convert the pandas DataFrame to a NumPy array
        y = self.y.values  # Convert the winner labels to a NumPy array
        return X, y