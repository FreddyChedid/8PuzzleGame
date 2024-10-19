import numpy as np
import heapq
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class EightPuzzle:
    def __init__(self, initial_state):
        self.state = np.array(initial_state)
        self.blank_pos = np.argwhere(self.state == 0)[0]

    def move(self, direction):
        """Move the blank tile in a given direction if possible."""
        row, col = self.blank_pos
        if direction == 'up' and row > 0:
            self._swap((row, col), (row - 1, col))
        elif direction == 'down' and row < 2:
            self._swap((row, col), (row + 1, col))
        elif direction == 'left' and col > 0:
            self._swap((row, col), (row, col - 1))
        elif direction == 'right' and col < 2:
            self._swap((row, col), (row, col + 1))

    def _swap(self, pos1, pos2):
        """Swap two positions in the puzzle."""
        self.state[tuple(pos1)], self.state[tuple(pos2)] = self.state[tuple(pos2)], self.state[tuple(pos1)]
        self.blank_pos = pos2  # Update the blank position

    def is_solved(self):
        """Check if the puzzle is solved."""
        return np.array_equal(self.state, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]]))

    def copy(self):
        """Return a copy of the current puzzle."""
        return EightPuzzle(self.state.copy())

    def legal_moves(self):
        """Returns a list of legal moves from the current state."""
        row, col = self.blank_pos
        moves = []
        if row > 0:
            moves.append('up')
        if row < 2:
            moves.append('down')
        if col > 0:
            moves.append('left')
        if col < 2:
            moves.append('right')
        return moves

    def result(self, move):
        """Returns the resulting state from applying a move."""
        new_puzzle = self.copy()
        new_puzzle.move(move)
        return new_puzzle

    def __eq__(self, other):
        return np.array_equal(self.state, other.state)

    def __hash__(self):
        return hash(self.state.tobytes())

    def __str__(self):
        """String representation of the puzzle state."""
        return '\n'.join([' '.join(map(str, row)) for row in self.state]) + '\n'


# A* Solver with Tie-Breaker
def a_star_solve(puzzle, heuristic_func):
    """Solve the 8-puzzle using A* and return the sequence of moves and the number of explored nodes."""
    frontier = []
    tie_breaker = 0  # Incremental counter to avoid comparing puzzles
    heapq.heappush(frontier, (0, tie_breaker, puzzle.copy(), []))  # Priority queue with (f_cost, tie_breaker, puzzle, moves)
    explored = set()
    explored_nodes = 0
    g_costs = {puzzle: 0}  # Cost to reach each state

    while frontier:
        f_cost, _, current_puzzle, moves = heapq.heappop(frontier)

        if current_puzzle.is_solved():
            return moves, explored_nodes

        if current_puzzle not in explored:
            explored.add(current_puzzle)
            explored_nodes += 1

            # Expand the current puzzle state
            for move in current_puzzle.legal_moves():
                new_puzzle = current_puzzle.result(move)
                new_g_cost = g_costs[current_puzzle] + 1  # Each move has a cost of 1

                if new_puzzle not in g_costs or new_g_cost < g_costs[new_puzzle]:
                    g_costs[new_puzzle] = new_g_cost
                    f_cost = new_g_cost + heuristic_func(new_puzzle)  # A* cost: f = g + h
                    tie_breaker += 1
                    heapq.heappush(frontier, (f_cost, tie_breaker, new_puzzle, moves + [move]))

    return [], explored_nodes  # No solution found


# Heuristic 1: Misplaced Tiles
def heuristic_misplaced_tiles(puzzle):
    goal = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    return np.sum(puzzle.state != goal) - 1  # Exclude the blank tile (0)

# Heuristic 2: Manhattan Distance
def heuristic_manhattan_distance(puzzle):
    goal_positions = {1: (0, 0), 2: (0, 1), 3: (0, 2),
                      4: (1, 0), 5: (1, 1), 6: (1, 2),
                      7: (2, 0), 8: (2, 1)}
    total_distance = 0
    for i in range(3):
        for j in range(3):
            tile = puzzle.state[i][j]
            if tile != 0:
                goal_pos = goal_positions[tile]
                total_distance += abs(i - goal_pos[0]) + abs(j - goal_pos[1])
    return total_distance

# Heuristic 3: Linear Conflict + Manhattan Distance
def heuristic_linear_conflict(puzzle):
    manhattan_dist = heuristic_manhattan_distance(puzzle)
    linear_conflicts = 0

    # Check row conflicts
    for row in range(3):
        current_row = puzzle.state[row]
        goal_row = [1 + 3 * row, 2 + 3 * row, 3 + 3 * row]
        for i in range(2):
            for j in range(i + 1, 3):
                if current_row[i] in goal_row and current_row[j] in goal_row and current_row[i] > current_row[j]:
                    linear_conflicts += 2

    # Check column conflicts
    for col in range(3):
        current_col = puzzle.state[:, col]
        goal_col = [1 + col, 4 + col, 7 + col]
        for i in range(2):
            for j in range(i + 1, 3):
                if current_col[i] in goal_col and current_col[j] in goal_col and current_col[i] > current_col[j]:
                    linear_conflicts += 2

    return manhattan_dist + linear_conflicts

# Heuristic 4: Zero Heuristic (Uniform Cost Search)
def heuristic_zero(puzzle):
    return 0


def update_display(puzzle, ax):
    """Update the puzzle display."""
    ax.clear()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)

    for i in range(3):
        for j in range(3):
            value = puzzle.state[2 - i][j]  # Reverse row index
            label = '' if value == 0 else str(value)
            ax.text(j + 0.5, i + 0.5, label, ha='center', va='center', fontsize=45, fontweight='bold',
                    bbox=dict(facecolor='lightgray' if value == 0 else 'white', 
                              edgecolor='black', boxstyle='round,pad=0.6', linewidth=2))

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)  # Adjust to fit the button
    plt.draw()

def on_click(event, puzzle, ax, solution_moves, step_counter):
    """Handle button click to go to the next move."""
    if step_counter[0] < len(solution_moves):
        move = solution_moves[step_counter[0]]
        puzzle.move(move)
        update_display(puzzle, ax)
        step_counter[0] += 1

def manual_animation_with_button(puzzle_initial_state, solution_moves):
    fig, ax = plt.subplots(figsize=(5, 5))

    # Create a button
    ax_next = plt.axes([0.8, 0.02, 0.15, 0.07])  # Position for button
    next_btn = Button(ax_next, 'Next')

    # Initialize the puzzle to the original initial state (resetting it)
    puzzle = EightPuzzle(puzzle_initial_state.copy())  # Reset puzzle to initial state

    # Initialize step counter for manual progression
    step_counter = [0]

    # Button callback
    next_btn.on_clicked(lambda event: on_click(event, puzzle, ax, solution_moves, step_counter))

    # Initial display of the puzzle
    update_display(puzzle, ax)
    plt.show()

import time

# Main execution logic
if __name__ == "__main__":
    # A solvable initial state
    initial_state = [[1, 3, 6],
                     [5, 4, 7],
                     [2, 0, 8]]  # The puzzle state

    # Run experiments with different heuristics
    print("Running A* with different heuristics...\n")
    for i, heuristic in enumerate([heuristic_misplaced_tiles, heuristic_manhattan_distance, heuristic_linear_conflict, heuristic_zero]):
        print(f"Experiment {i+1}: Using Heuristic {heuristic.__name__}")
        puzzle = EightPuzzle(initial_state.copy())

        start_time = time.time()
        solution_moves, explored_nodes = a_star_solve(puzzle, heuristic)
        end_time = time.time()
        time_taken = end_time - start_time

        if solution_moves:
            print(f"Number of explored nodes: {explored_nodes}")
            print(f"Solution path (moves): {solution_moves}")
            print(f"Time taken: {time_taken:.4f} seconds")  # Print the time taken
            manual_animation_with_button(initial_state, solution_moves)  # Display the solution
        else:
            print("No solution found or the puzzle is unsolvable.")
        print()
