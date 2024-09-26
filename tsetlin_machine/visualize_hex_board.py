import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import numpy as np
import matplotlib.patches as mpatches

def visualize_hex_board(board_state, board_size=7, output_file='hex_board.png'):
    """
    Visualizes a Hex board state as a rhombus with sides labeled for each player.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')

    # Define colors for players and empty cells
    color_map = {
        -1: '#1f77b4',  # Player 2 (Blue)
        0: '#ffffff',   # Empty cell
        1: '#d62728'    # Player 1 (Red)
    }

    # Hexagon parameters
    hex_size = 1  # Size of each hexagon

    # Collect x and y coordinates for adjusting plot limits
    all_x_coords = []
    all_y_coords = []

    # Centering offsets
    center_offset = (board_size - 1) / 2

    # Loop over board positions to create hexagons
    for row in range(board_size):
        for col in range(board_size):
            # Axial coordinates (q, r)
            flipped_col = board_size - 1 - col
            q = (row - center_offset)
            r = (flipped_col - center_offset)

            # Convert axial coordinates to pixel coordinates
            x = hex_size * np.sqrt(3) * (q - r / 2)
            y = hex_size * 3/2 * r


            all_x_coords.append(x)
            all_y_coords.append(y)

            cell_value = board_state[f'cell{row}_{col}']
            color = color_map[cell_value]

            # Create a hexagon at the calculated position
            hexagon = RegularPolygon(
                (x, y),
                numVertices=6,
                radius=hex_size * 0.95,  # Slight reduction to avoid overlaps
                orientation=np.radians(0),  # Pointy-top hexagons
                facecolor=color,
                edgecolor='gray'
            )
            ax.add_patch(hexagon)

    # Adjust plot limits
    x_min = min(all_x_coords) - hex_size * np.sqrt(3)
    x_max = max(all_x_coords) + hex_size * np.sqrt(3)
    y_min = min(all_y_coords) - hex_size * 1.5
    y_max = max(all_y_coords) + hex_size * 1.5

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Remove axes
    ax.axis('off')

    # Add title
    ax.set_title('Hex Game Board', fontsize=20)

    # Create legend patches
    legend_patches = [
        mpatches.Patch(color=color_map[1], label='Player 1 (Red)'),
        mpatches.Patch(color=color_map[-1], label='Player 2 (Blue)'),
        mpatches.Patch(facecolor=color_map[0], edgecolor='gray', label='Empty Cell')
    ]

    # Add the legend outside the plot
    ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=14)

    # Highlight the sides for each player
    side_margin = hex_size * 0.2  # Adjust as needed

    # Coordinates for Player 1's sides (Red)
    red_side_coords = [
        [(x_min - side_margin, y_min - side_margin), (x_max + side_margin, y_min - side_margin)],  # Bottom side
        [(x_min - side_margin, y_max + side_margin), (x_max + side_margin, y_max + side_margin)]   # Top side
    ]

    # Coordinates for Player 2's sides (Blue)
    blue_side_coords = [
        [(x_min - side_margin, y_min - side_margin), (x_min - side_margin, y_max + side_margin)],  # Left side
        [(x_max + side_margin, y_min - side_margin), (x_max + side_margin, y_max + side_margin)]   # Right side
    ]

    # Add lines or shaded areas to represent the sides
    for coords in red_side_coords:
        line = plt.Line2D(
            (coords[0][0], coords[1][0]),
            (coords[0][1], coords[1][1]),
            color=color_map[1],
            linewidth=5
        )
        ax.add_line(line)

    for coords in blue_side_coords:
        line = plt.Line2D(
            (coords[0][0], coords[1][0]),
            (coords[0][1], coords[1][1]),
            color=color_map[-1],
            linewidth=5
        )
        ax.add_line(line)

    # Optionally, add labels to the sides
    # For Player 1 (Red)
    ax.text((x_min + x_max) / 2, y_max + hex_size * 1.5, 'Player 1 Side', color=color_map[1],
            ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.text((x_min + x_max) / 2, y_min - hex_size * 1.5, 'Player 1 Side', color=color_map[1],
            ha='center', va='top', fontsize=12, fontweight='bold')

    # For Player 2 (Blue)
    ax.text(x_min - hex_size * 1.5, (y_min + y_max) / 2, 'Player 2 Side', color=color_map[-1],
            ha='right', va='center', fontsize=12, fontweight='bold', rotation=90)
    ax.text(x_max + hex_size * 1.5, (y_min + y_max) / 2, 'Player 2 Side', color=color_map[-1],
            ha='left', va='center', fontsize=12, fontweight='bold', rotation=90)

    # Save the figure to a file
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free up memory

    print(f"Board visualization saved as {output_file}")