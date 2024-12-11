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

    def load_data(self, desired_samples_per_class, balance_classes=True):
        """
        Loads the dataset from the CSV file and balances the classes to have the desired number of samples per class.
        
        Parameters:
        - desired_samples_per_class: The specific number of samples to load per class.
        - balance_classes: Must be True. Ensures that the dataset has an equal number of samples for each class.
        """
        if not balance_classes:
            raise ValueError("balance_classes must be True when specifying desired_samples_per_class.")
        
        # Initialize empty lists to collect samples
        class_positive_samples = []
        class_negative_samples = []
        
        # Initialize counters
        pos_samples_collected = 0
        neg_samples_collected = 0
        
        # Read the CSV file in chunks to handle large datasets
        chunksize = 10000  # Adjust based on memory capacity
        for chunk in pd.read_csv(self.data_file, chunksize=chunksize):
            
            chunk = chunk.sample(frac=1, random_state=42).reset_index(drop=True)

            # Separate the chunk into classes
            pos_chunk = chunk[chunk['winner'] == 1]
            neg_chunk = chunk[chunk['winner'] == -1]
            
            # Calculate how many samples to take from this chunk
            pos_needed = desired_samples_per_class - pos_samples_collected
            neg_needed = desired_samples_per_class - neg_samples_collected
            
            # Take samples from positive class
            if pos_needed > 0 and not pos_chunk.empty:
                pos_to_take = min(pos_needed, len(pos_chunk))
                pos_samples = pos_chunk.sample(n=pos_to_take, random_state=42)
                class_positive_samples.append(pos_samples)
                pos_samples_collected += pos_to_take
            
            # Take samples from negative class
            if neg_needed > 0 and not neg_chunk.empty:
                neg_to_take = min(neg_needed, len(neg_chunk))
                neg_samples = neg_chunk.sample(n=neg_to_take, random_state=42)
                class_negative_samples.append(neg_samples)
                neg_samples_collected += neg_to_take
            
            # Check if we have collected enough samples
            if pos_samples_collected >= desired_samples_per_class and neg_samples_collected >= desired_samples_per_class:
                break
        
        # Handle the case where not enough samples are available
        if pos_samples_collected < desired_samples_per_class or neg_samples_collected < desired_samples_per_class:
            print(f"Warning: Not enough samples collected. Collected {pos_samples_collected} positive and {neg_samples_collected} negative samples.")
            # Adjust the desired samples per class to the minimum collected
            desired_samples_per_class = min(pos_samples_collected, neg_samples_collected)
            # Truncate the collected samples to the new desired count
            class_positive_samples = [df.head(desired_samples_per_class) for df in class_positive_samples]
            class_negative_samples = [df.head(desired_samples_per_class) for df in class_negative_samples]
        
        # Combine the samples from all chunks
        class_positive_sampled = pd.concat(class_positive_samples)
        class_negative_sampled = pd.concat(class_negative_samples)
        
        # Ensure equal number of samples per class
        class_positive_sampled = class_positive_sampled.head(desired_samples_per_class)
        class_negative_sampled = class_negative_sampled.head(desired_samples_per_class)
        
        # Combine the sampled DataFrames
        self.data = pd.concat([class_positive_sampled, class_negative_sampled])
        
        # Shuffle the combined DataFrame
        self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)
        
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
        board_state = self.X.iloc[idx]
        winner = self.y.iloc[idx]
        return board_state, winner

    def get_all_data(self):
        """
        Retrieves the entire dataset in a format ready for model input.
        
        Returns:
        - X: The board states as a NumPy array.
        - y: The game winners as a NumPy array.
        """
        X = self.X.values
        y = self.y.values
        return X, y
