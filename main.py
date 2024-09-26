from tsetlin_machine.hex_dataset_loader import HexDataLoader
from tsetlin_machine.visualize_hex_board import visualize_hex_board

# Example usage
if __name__ == '__main__':
    # Initialize the data loader with your dataset path and board size
    data_loader = HexDataLoader('hex_winning_positions.csv', board_size=5)

    # Load the data
    data_loader.load_data()

    # Get a random game state and winner
    board_state, winner = data_loader.get_random_entry()
    print(board_state)
    print(winner)

    print(f'Winner of this game: {"Player 1" if winner == 1 else "Player 2"}')

    # Visualize the board state and save it as an image
    visualize_hex_board(board_state, board_size=data_loader.board_size, output_file='hex_board.png')