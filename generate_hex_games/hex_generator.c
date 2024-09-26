#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef BOARD_DIM
#define BOARD_DIM 5
#endif

// Neighboring cell offsets for hexagonal grid.
int neighbors[] = {-(BOARD_DIM + 2) + 1, -(BOARD_DIM + 2), -1, 1, (BOARD_DIM + 2), (BOARD_DIM + 2) - 1};

struct hex_game {
    int board[(BOARD_DIM + 2) * (BOARD_DIM + 2)];
    int open_positions[BOARD_DIM * BOARD_DIM];
    int number_of_open_positions;
};

// Initialize the hex game with empty cells and open positions.
void hg_init(struct hex_game *hg) {
    for (int i = 0; i < BOARD_DIM + 2; ++i) {
        for (int j = 0; j < BOARD_DIM + 2; ++j) {
            hg->board[i * (BOARD_DIM + 2) + j] = 0;
            if (i > 0 && i < BOARD_DIM + 1 && j > 0 && j < BOARD_DIM + 1) {
                hg->open_positions[(i - 1) * BOARD_DIM + j - 1] = i * (BOARD_DIM + 2) + j;
            }
        }
    }
    hg->number_of_open_positions = BOARD_DIM * BOARD_DIM;
}

// Recursive DFS function to check connections for a player.
int hg_dfs(struct hex_game *hg, int player, int position, int *visited) {
    visited[position] = 1;  // Mark this position as visited.
    int row = position / (BOARD_DIM + 2);
    int col = position % (BOARD_DIM + 2);

    // Check if player 1 reached the right edge or player -1 reached the bottom edge.
    if ((player == 1 && col == BOARD_DIM) || (player == -1 && row == BOARD_DIM)) {
        return 1;
    }

    // Explore all neighbors.
    for (int i = 0; i < 6; ++i) {
        int neighbor = position + neighbors[i];
        if (neighbor >= 0 && neighbor < (BOARD_DIM + 2) * (BOARD_DIM + 2) &&
            hg->board[neighbor] == player && !visited[neighbor]) {
            if (hg_dfs(hg, player, neighbor, visited)) {
                return 1;
            }
        }
    }

    return 0;
}

// Function to check if a player has won.
int hg_winner(struct hex_game *hg, int player) {
    int visited[(BOARD_DIM + 2) * (BOARD_DIM + 2)];
    memset(visited, 0, sizeof(visited));  // Reset visited array.

    if (player == 1) {
        // Check if player 1 has connected left to right.
        for (int row = 1; row <= BOARD_DIM; ++row) {
            int start_pos = row * (BOARD_DIM + 2) + 1;  // Starting from the left edge.
            if (hg->board[start_pos] == 1 && hg_dfs(hg, 1, start_pos, visited)) {
                return 1;  // Player 1 wins.
            }
        }
    } else {
        // Check if player -1 has connected top to bottom.
        for (int col = 1; col <= BOARD_DIM; ++col) {
            int start_pos = (BOARD_DIM + 2) + col;  // Starting from the top edge.
            if (hg->board[start_pos] == -1 && hg_dfs(hg, -1, start_pos, visited)) {
                return -1;  // Player -1 wins.
            }
        }
    }

    return 0;
}

// Place a piece randomly on the board.
int hg_place_piece_randomly(struct hex_game *hg, int player) {
    int random_empty_position_index = rand() % hg->number_of_open_positions;
    int empty_position = hg->open_positions[random_empty_position_index];
    hg->board[empty_position] = player;
    hg->open_positions[random_empty_position_index] = hg->open_positions[hg->number_of_open_positions - 1];
    hg->number_of_open_positions--;
    return empty_position;
}

// Check if the board is full.
int hg_full_board(struct hex_game *hg) {
    return hg->number_of_open_positions == 0;
}

// Save the game state and winner to a CSV file.
void save_game_to_csv(struct hex_game *hg, int winner, FILE *file) {
    for (int col = 1; col <= BOARD_DIM; ++col) {
        for (int row = 1; row <= BOARD_DIM; ++row) {
            int value = hg->board[row * (BOARD_DIM + 2) + col];
            fprintf(file, "%d,", value);
        }
    }
    fprintf(file, "%d\n", winner);
}

// Write the CSV headers.
void write_csv_headers(FILE *file) {
    for (int col = 0; col < BOARD_DIM; ++col) {
        for (int row = 0; row < BOARD_DIM; ++row) {
            fprintf(file, "cell%d_%d,", col, row);
        }
    }
    fprintf(file, "winner\n");
}

int main() {
    struct hex_game hg;
    FILE *csv_file = fopen("hex_games_11.csv", "w");
    if (csv_file == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    write_csv_headers(csv_file);

    for (int game = 0; game < 1000; ++game) {
        hg_init(&hg);
        int player = 1;  // Player 1 starts (O).
        int winner = 0;

        while (!hg_full_board(&hg)) {
            int position = hg_place_piece_randomly(&hg, player);
            winner = hg_winner(&hg, player);
            if (winner != 0) {
                break;
            }
            player = -player;  // Switch player.
        }

        save_game_to_csv(&hg, winner, csv_file);

        if (game % 100 == 0) {
            printf("Game %d completed.\n", game);
        }
    }

    fclose(csv_file);
    return 0;
}
