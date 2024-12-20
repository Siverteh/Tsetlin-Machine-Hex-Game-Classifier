from tsetlin_machine.hex_dataset_loader import HexDataLoader
from tsetlin_machine.visualize_hex_board import visualize_hex_board

# Example usage
if __name__ == '__main__':
    # Initialize the data loader with your dataset path and board size
    data_loader = HexDataLoader('C:/Users/Eier/Tsetlin-Machine-Hex-Game-Classifier/datasets/hex_games_five_moves_ahead.csv', board_size=7)

    # Load the data
    data_loader.load_data(desired_samples_per_class=10)

    # Get a random game state and winner
    board_state, winner = data_loader.get_random_entry()
    print(board_state)
    print(winner)

    print(f'Winner of this game: {"Player 1" if winner == 1 else "Player 2"}')

    # Visualize the board state and save it as an image
    visualize_hex_board(board_state, board_size=data_loader.board_size, output_file='hex_board.png')