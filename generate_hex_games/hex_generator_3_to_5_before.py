import random
import csv
import os
import multiprocessing
from multiprocessing import Pool, Manager, Lock

BOARD_DIM = 5  # Board dimension (5x5)

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
        self.moves = []
        self.init_board()

    def init_board(self):
        self.open_positions = []
        for i in range(1, BOARD_DIM + 1):
            for j in range(1, BOARD_DIM + 1):
                self.open_positions.append(i * (BOARD_DIM + 2) + j)

    def place_piece(self, player, position):
        if position not in self.open_positions:
            return False
        self.board[position] = player
        self.moves.append((position, player))
        self.open_positions.remove(position)
        return True

    def remove_piece(self, position):
        self.board[position] = 0
        self.open_positions.append(position)
        self.moves = [(pos, p) for pos, p in self.moves if pos != position]

    def dfs(self, player, position, visited):
        stack = [position]
        while stack:
            pos = stack.pop()
            if visited[pos]:
                continue
            visited[pos] = True
            row = pos // (BOARD_DIM + 2)
            col = pos % (BOARD_DIM + 2)
            if player == 1 and col == BOARD_DIM:
                return True
            if player == -1 and row == BOARD_DIM:
                return True
            for neighbor_offset in neighbors:
                neighbor = pos + neighbor_offset
                if 0 <= neighbor < len(self.board) and not visited[neighbor]:
                    if self.board[neighbor] == player:
                        stack.append(neighbor)
        return False

    def check_guaranteed_win(self, player):
        visited = [False] * len(self.board)
        if player == 1:
            # Player 1 wins left-right
            for row in range(1, BOARD_DIM + 1):
                start_pos = row*(BOARD_DIM+2)+1
                if self.board[start_pos] == player and not visited[start_pos]:
                    if self.dfs(player, start_pos, visited):
                        return True
        else:
            # Player -1 wins top-bottom
            for col in range(1, BOARD_DIM + 1):
                start_pos = (BOARD_DIM+2)+col
                if self.board[start_pos] == player and not visited[start_pos]:
                    if self.dfs(player, start_pos, visited):
                        return True
        return False

    def get_board_state(self):
        board_state = []
        for i in range(1, BOARD_DIM + 1):
            row = self.board[i*(BOARD_DIM+2)+1 : i*(BOARD_DIM+2)+BOARD_DIM+1]
            board_state.extend(row)
        return board_state

    def clone(self):
        new_game = HexGame()
        new_game.board = self.board[:]
        new_game.open_positions = self.open_positions[:]
        new_game.moves = self.moves[:]
        return new_game

memo = {}

def can_force_win_in_n_moves(game, current_player, target_winner, moves_left):
    """
    Returns True if 'target_winner' can force a win in exactly 'moves_left' moves 
    under perfect play from both sides.
    """
    state_key = (tuple(game.get_board_state()), current_player, target_winner, moves_left)
    if state_key in memo:
        return memo[state_key]

    if moves_left == 0:
        result = game.check_guaranteed_win(target_winner)
        memo[state_key] = result
        return result

    if current_player == target_winner:
        # Target winner's turn: at least one move leads to forced win
        any_forced = False
        for pos in game.open_positions:
            game.place_piece(current_player, pos)
            if can_force_win_in_n_moves(game, -current_player, target_winner, moves_left - 1):
                any_forced = True
            game.remove_piece(pos)
            if any_forced:
                break
        memo[state_key] = any_forced
        return any_forced
    else:
        # Opponent's turn: all moves must preserve forced win
        all_forced = True
        for opp_pos in game.open_positions:
            game.place_piece(current_player, opp_pos)
            if not can_force_win_in_n_moves(game, -current_player, target_winner, moves_left - 1):
                all_forced = False
            game.remove_piece(opp_pos)
            if not all_forced:
                break
        memo[state_key] = all_forced
        return all_forced

def no_forced_win_before_n(game, winning_player, n):
    # Ensure no forced win at fewer than n moves
    for x in range(1, n):
        memo.clear()
        if can_force_win_in_n_moves(game.clone(), winning_player, winning_player, x):
            return False
    return True

def exactly_n_moves_forced_win(game, winning_player, n=5):
    # Clear memo before checks
    memo.clear()
    # Not already won
    if game.check_guaranteed_win(1) or game.check_guaranteed_win(-1):
        return False
    # No forced win before n moves
    if not no_forced_win_before_n(game.clone(), winning_player, n):
        return False
    # Forced win at exactly n moves
    memo.clear()
    if not can_force_win_in_n_moves(game.clone(), winning_player, winning_player, n):
        return False
    # Double-check no forced win before n moves
    if not no_forced_win_before_n(game.clone(), winning_player, n):
        return False
    return True

def generate_game_state_with_forced_win(seed):
    random.seed(seed)
    game = HexGame()
    move_count = 0
    current_player = 1
    move_limit = random.randint(5, BOARD_DIM * BOARD_DIM - 5)
    while move_count < move_limit and game.open_positions:
        pos = random.choice(game.open_positions)
        game.place_piece(current_player, pos)
        move_count += 1
        current_player = -current_player

    # Set winning_player to the next to move, ensuring they start the 5-move sequence
    winning_player = current_player

    # Turn parity check to avoid invalid states
    player1_moves = sum(1 for _, p in game.moves if p == 1)
    player_neg1_moves = sum(1 for _, p in game.moves if p == -1)
    if player1_moves < player_neg1_moves or player1_moves > player_neg1_moves + 1:
        return None

    # Check if exactly 5 moves away from forced win
    if exactly_n_moves_forced_win(game, winning_player, n=5):
        board_state = tuple(game.get_board_state())
        return board_state, winning_player
    return None

def write_csv_headers(csv_writer):
    headers = []
    for row in range(BOARD_DIM):
        for col in range(BOARD_DIM):
            headers.append(f"cell{row}_{col}")
    headers.append("winner")
    csv_writer.writerow(headers)

def append_game_to_csv(lock, filename, board_state, winner, shared_states, shared_count):
    with lock:
        # Ensure global uniqueness
        if board_state in shared_states:
            return
        shared_states[board_state] = True

        file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
        with open(filename, mode='a', newline='') as file:
            csv_writer = csv.writer(file)
            # Write headers only if file is empty
            if not file_exists:
                write_csv_headers(csv_writer)
            csv_writer.writerow(list(board_state) + [winner])

        # Increment the global counter
        shared_count.value += 1
        if shared_count.value % 1000 == 0:
            print(f"{shared_count.value} board states have been saved to the CSV.")

def worker_generate_games(num_games_per_worker, worker_id, filename, lock, shared_states, shared_count):
    seed = random.randint(0, 1_000_000_000) + worker_id
    attempts = 0
    found = 0
    while found < num_games_per_worker and attempts < num_games_per_worker * 1000:
        result = generate_game_state_with_forced_win(seed)
        seed += 1
        attempts += 1
        if result:
            board_state, winner = result
            # Append game to CSV if not a duplicate
            if board_state not in shared_states:
                append_game_to_csv(lock, filename, board_state, winner, shared_states, shared_count)
                found += 1
    return found

if __name__ == "__main__":
    import time

    total_unique_games = 100  # Number of total games to generate
    num_workers = multiprocessing.cpu_count()
    num_games_per_worker = total_unique_games // num_workers + 1

    filename = 'hex_games_exactly_five_moves_ahead.csv'

    manager = Manager()
    lock = manager.Lock()
    shared_states = manager.dict()
    shared_count = manager.Value('i', 0)

    print(f"Using {num_workers} worker processes.")
    start_time = time.time()

    with Pool(processes=num_workers) as pool:
        worker_args = [(num_games_per_worker, i, filename, lock, shared_states, shared_count) for i in range(num_workers)]
        results = pool.starmap(worker_generate_games, worker_args)

    total_found = sum(results)
    end_time = time.time()
    print(f"Generated {total_found} unique games in {end_time - start_time:.2f} seconds.")
