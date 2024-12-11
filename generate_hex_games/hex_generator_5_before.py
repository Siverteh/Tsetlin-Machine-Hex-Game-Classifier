import random
import csv
import multiprocessing
from multiprocessing import Pool
import itertools
from functools import partial

BOARD_DIM = 5  # Board dimension (5x5)

# Neighbor offsets for a hexagonal grid
neighbors = [
    -(BOARD_DIM + 2) + 1,  # Up-Right
    -(BOARD_DIM + 2),      # Up-Left
    -1,                    # Left
    1,                     # Right
    (BOARD_DIM + 2),       # Down-Right
    (BOARD_DIM + 2) - 1    # Down-Left
]

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

    def place_piece(self, player, position):
        """Place a piece at the given position."""
        if position not in self.open_positions:
            return False
        self.board[position] = player
        self.moves.append((position, player))
        self.open_positions.remove(position)
        return True

    def remove_piece(self, position):
        """Remove a piece from the given position."""
        self.board[position] = 0
        self.open_positions.append(position)
        self.moves = [(pos, p) for pos, p in self.moves if pos != position]

    def dfs(self, player, position, visited):
        """DFS to explore connectivity for the given player."""
        stack = [position]
        while stack:
            pos = stack.pop()
            if visited[pos]:
                continue
            visited[pos] = True
            row = pos // (BOARD_DIM + 2)
            col = pos % (BOARD_DIM + 2)
            # Player 1 (X) wins by connecting left to right
            if player == 1 and col == BOARD_DIM:
                return True
            # Player -1 (O) wins by connecting top to bottom
            if player == -1 and row == BOARD_DIM:
                return True
            # Explore neighbors
            for neighbor_offset in neighbors:
                neighbor = pos + neighbor_offset
                if 0 <= neighbor < len(self.board) and not visited[neighbor]:
                    if self.board[neighbor] == player:
                        stack.append(neighbor)
        return False

    def check_guaranteed_win(self, player):
        """Check if the player has a guaranteed winning position."""
        visited = [False] * len(self.board)
        if player == 1:
            # Player 1 (X) must connect left to right
            for row in range(1, BOARD_DIM + 1):
                start_pos = row * (BOARD_DIM + 2) + 1  # Leftmost column
                if self.board[start_pos] == player and not visited[start_pos]:
                    if self.dfs(player, start_pos, visited):
                        return True
        else:
            # Player -1 (O) must connect top to bottom
            for col in range(1, BOARD_DIM + 1):
                start_pos = (BOARD_DIM + 2) + col  # Top row
                if self.board[start_pos] == player and not visited[start_pos]:
                    if self.dfs(player, start_pos, visited):
                        return True
        return False

    def get_board_state(self):
        """Return the board state in a row-major order for CSV saving."""
        board_state = []
        for i in range(1, BOARD_DIM + 1):
            row = self.board[i * (BOARD_DIM + 2) + 1: i * (BOARD_DIM + 2) + BOARD_DIM + 1]
            board_state.extend(row)
        return board_state

    def clone(self):
        """Create a deep copy of the game."""
        new_game = HexGame()
        new_game.board = self.board[:]
        new_game.open_positions = self.open_positions[:]
        new_game.moves = self.moves[:]
        return new_game

def has_forced_win_in_n_moves(game, winning_player, moves_left, visited=None):
    """Check if the winning player has an unavoidable win in 'moves_left' moves."""
    if moves_left == 0:
        # Base case: Check if it's a guaranteed win now
        return game.check_guaranteed_win(winning_player)

    if visited is None:
        visited = set()

    current_player = -winning_player  # Opponent's turn

    for opp_move in game.open_positions:
        # Avoid revisiting the same state
        if opp_move in visited:
            continue
        visited.add(opp_move)

        game.place_piece(current_player, opp_move)

        # Winning player tries to win after opponent's move
        win_found = False
        for win_move in game.open_positions:
            if win_move == opp_move:
                continue
            game.place_piece(winning_player, win_move)
            if has_forced_win_in_n_moves(game, winning_player, moves_left - 1, visited):
                win_found = True
            game.remove_piece(win_move)
            if win_found:
                break  # Stop if a winning path is found

        game.remove_piece(opp_move)
        if not win_found:
            # If the opponent can block all winning paths, no forced win
            return False

    return True  # Winning player has an unavoidable win



def generate_game_state_with_forced_win(seed, moves_away=5):
    """Generate a game state where one player has a forced win in 'moves_away' moves."""
    random.seed(seed)
    game = HexGame()
    move_count = 0
    current_player = 1  # Player 1 starts
    move_limit = random.randint(5, BOARD_DIM * BOARD_DIM - 5)  # Random number of moves to simulate

    while move_count < move_limit and game.open_positions:
        pos = random.choice(game.open_positions)
        game.place_piece(current_player, pos)
        move_count += 1
        current_player = -current_player

    # It's now current_player's turn (since we switched after last move)
    # So the winning player is -current_player
    winning_player = -current_player

    # Check move counts are correct
    player1_moves = sum(1 for _, p in game.moves if p == 1)
    player_neg1_moves = sum(1 for _, p in game.moves if p == -1)
    if player1_moves < player_neg1_moves or player1_moves > player_neg1_moves + 1:
        # Invalid move counts, try again
        return None

    if has_forced_win_in_n_moves(game, winning_player, moves_away):
        board_state = tuple(game.get_board_state())
        return board_state, winning_player
    else:
        return None

def save_games_to_csv(games, filename):
    """Save the list of games to a CSV file."""
    with open(filename, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        write_csv_headers(csv_writer)
        for board_state, winner in games:
            csv_writer.writerow(list(board_state) + [winner])

def write_csv_headers(csv_writer):
    """Write the CSV headers in the format cell{row}_{col} and winner."""
    headers = []
    for row in range(BOARD_DIM):
        for col in range(BOARD_DIM):
            headers.append(f"cell{row}_{col}")
    headers.append("winner")
    csv_writer.writerow(headers)

def worker_generate_games(num_games_per_worker, worker_id, moves_away):
    """Worker function to generate games."""
    unique_games = set()
    games = []
    seed = random.randint(0, 1_000_000_000) + worker_id
    attempts = 0
    while len(unique_games) < num_games_per_worker and attempts < num_games_per_worker * 10:
        result = generate_game_state_with_forced_win(seed, moves_away)
        seed += 1  # Change seed for the next iteration
        attempts += 1
        if result:
            board_state, winner = result
            if board_state not in unique_games:
                unique_games.add(board_state)
                games.append((board_state, winner))
    return games

if __name__ == "__main__":
    import time

    total_unique_games = 1000  # Total number of unique games to generate
    num_workers = multiprocessing.cpu_count()  # Use all available CPU cores
    num_games_per_worker = total_unique_games // num_workers + 1
    moves_away = 5  # Number of moves away from winning

    print(f"Using {num_workers} worker processes to generate games {moves_away} moves away from winning.")

    start_time = time.time()

    with Pool(processes=num_workers) as pool:
        # Map the worker function to the number of workers
        worker_args = [(num_games_per_worker, i, moves_away) for i in range(num_workers)]
        # Use starmap to pass multiple arguments
        results = pool.starmap(worker_generate_games, worker_args)

    # Combine results from all workers
    combined_games = []
    unique_game_states = set()
    for games in results:
        for board_state, winner in games:
            if board_state not in unique_game_states:
                unique_game_states.add(board_state)
                combined_games.append((board_state, winner))

    # Save to CSV
    save_games_to_csv(combined_games[:total_unique_games], f'hex_games_{moves_away}_moves_away_{BOARD_DIM}.csv')

    end_time = time.time()
    print(f"Generated {len(combined_games)} unique games in {end_time - start_time:.2f} seconds.")