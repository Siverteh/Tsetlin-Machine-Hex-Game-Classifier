import random
import csv

BOARD_DIM = 5

# Neighbor offsets for a hexagonal grid
neighbors = [-(BOARD_DIM + 2) + 1, -(BOARD_DIM + 2), -1, 1, (BOARD_DIM + 2), (BOARD_DIM + 2) - 1]

class HexGame:
    def __init__(self):
        self.board = [0] * ((BOARD_DIM + 2) * (BOARD_DIM + 2))
        self.open_positions = []
        self.moves = []  # Track the moves to allow for rewinding
        self.init_board()

    def init_board(self):
        """Initialize the board and open positions."""
        self.open_positions = []
        for i in range(1, BOARD_DIM + 1):
            for j in range(1, BOARD_DIM + 1):
                self.open_positions.append(i * (BOARD_DIM + 2) + j)

    def place_piece_randomly(self, player):
        """Place a piece randomly on the board."""
        if not self.open_positions:
            return None

        random_empty_position = random.choice(self.open_positions)
        self.board[random_empty_position] = player
        self.moves.append((random_empty_position, player))  # Track the move
        self.open_positions.remove(random_empty_position)
        return random_empty_position

    def remove_last_move(self):
        """Remove the last move to rewind the game."""
        if not self.moves:
            return
        last_move, player = self.moves.pop()
        self.board[last_move] = 0  # Remove the last piece
        self.open_positions.append(last_move)  # Restore the open position

    def rewind_moves(self, num_moves):
        """Rewind the game by a specific number of moves."""
        for _ in range(num_moves):
            if self.moves:
                self.remove_last_move()

    def dfs(self, player, position, visited):
        """DFS to explore connectivity for the given player."""
        visited[position] = True
        row = position // (BOARD_DIM + 2)
        col = position % (BOARD_DIM + 2)

        # Player 1 (O) wins by connecting top to bottom
        if player == 1 and row == BOARD_DIM:
            return True

        # Player -1 (X) wins by connecting left to right
        if player == -1 and col == BOARD_DIM:
            return True

        # Explore neighbors
        for neighbor_offset in neighbors:
            neighbor = position + neighbor_offset
            if 0 <= neighbor < (BOARD_DIM + 2) * (BOARD_DIM + 2) and not visited[neighbor]:
                if self.board[neighbor] == player:
                    if self.dfs(player, neighbor, visited):
                        return True
        return False

    def check_guaranteed_win(self, player):
        """Check if the player is in a guaranteed winning position."""
        visited = [False] * ((BOARD_DIM + 2) * (BOARD_DIM + 2))

        if player == 1:
            # Player 1 must connect top to bottom
            for col in range(1, BOARD_DIM + 1):
                start_pos = (BOARD_DIM + 2) + col  # Top row
                if self.board[start_pos] == player and not visited[start_pos]:
                    if self.dfs(player, start_pos, visited):
                        return True
        else:
            # Player -1 must connect left to right
            for row in range(1, BOARD_DIM + 1):
                start_pos = row * (BOARD_DIM + 2) + 1  # Leftmost column
                if self.board[start_pos] == player and not visited[start_pos]:
                    if self.dfs(player, start_pos, visited):
                        return True
        return False

    def full_board(self):
        """Check if the board is full."""
        return len(self.open_positions) == 0

    def get_board_state(self):
        """Return the board state in a column-major order for CSV saving."""
        board_state = []
        for col in range(1, BOARD_DIM + 1):
            for row in range(1, BOARD_DIM + 1):
                board_state.append(self.board[row * (BOARD_DIM + 2) + col])
        return board_state

def save_game_to_csv(game, winner, csv_writer):
    """Save the board state and the winner to a CSV file."""
    board_state = game.get_board_state()
    csv_writer.writerow(board_state + [winner])

def write_csv_headers(csv_writer):
    """Write the CSV headers in the format cell{col}_{row} and winner."""
    headers = []
    for col in range(BOARD_DIM):
        for row in range(BOARD_DIM):
            headers.append(f"cell{col}_{row}")
    headers.append("winner")
    csv_writer.writerow(headers)

def simulate_hex_game(game):
    """Simulate a Hex game until one player wins."""
    current_player = 1  # Player 1 starts (O)

    while not game.full_board():
        game.place_piece_randomly(current_player)
        if game.check_guaranteed_win(current_player):
            return current_player  # Return the winner as soon as they have a guaranteed victory

        current_player = -current_player  # Switch player

    return 0  # Shouldn't happen in a complete game

# Simulate and save games to CSV
if __name__ == "__main__":
    num_games_per_category = 1000 // 6  # 1000 total games, equally divided across categories
    player_1_wins = 0
    player_neg1_wins = 0

    with open('hex_winning_positions.csv', mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        write_csv_headers(csv_writer)

        for rewind_moves in range(0, 6):  # 0 = fully connected, 1 to 5 moves away
            for game_num in range(num_games_per_category):
                game = HexGame()

                winner = simulate_hex_game(game)  # Simulate the full game
                if rewind_moves > 0:
                    # Rewind 1 to 5 moves to create near-winning states
                    game.rewind_moves(rewind_moves)

                # Save the game result to the CSV
                save_game_to_csv(game, winner, csv_writer)

                if winner == 1:
                    player_1_wins += 1
                elif winner == -1:
                    player_neg1_wins += 1

                if game_num % 100 == 0:
                    print(f"Rewind {rewind_moves} - Game {game_num} completed.")

    print(f"Player 1 wins: {player_1_wins}")
    print(f"Player -1 wins: {player_neg1_wins}")
