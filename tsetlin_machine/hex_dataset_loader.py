import pandas as pd
import random
from tsetlin_machine.visualize_hex_board import visualize_hex_board

class HexDataLoader:
    """
    A class to load and handle the Hex game dataset.
    """
    def __init__(self, data_file, board_size=7):
        """
        Initializes the data loader with the dataset file path and board size.
        """
        self.data_file = data_file
        self.board_size = board_size
        self.data = None
        self.X = None
        self.y = None

    def load_data(self):
        """
        Loads the dataset from the CSV file.
        """
        # Read the CSV file into a pandas DataFrame
        self.data = pd.read_csv(self.data_file)

        # Separate features and labels
        self.X = self.data.drop('winner', axis=1)
        self.y = self.data['winner']

    def get_random_entry(self):
        """
        Retrieves a random game state and its winner.
        """
        idx = random.randint(0, len(self.X) - 1)
        board_state = self.X.iloc[idx]
        winner = self.y.iloc[idx]
        return board_state, winner



