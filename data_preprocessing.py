import chess.pgn
import pandas as pd
import time
import os

def parse_pgn(file_path, output_file):
    games = []
    start_time = time.time()  # Track start time

    # Create the output file and write the header
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write("fen,move\n")  # Write CSV header

    # Open the PGN file
    with open(file_path) as pgn:
        game_count = 0
        move_count = 0

        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            game_count += 1
            board = game.board()

            # Write each move to the file immediately
            with open(output_file, "a") as f:
                for move in game.mainline_moves():
                    move_count += 1
                    fen = board.fen()
                    move_uci = move.uci()

                    # Write the move to the file
                    f.write(f"{fen},{move_uci}\n")

                    # Update the board state
                    board.push(move)

                    # Show progress in the terminal
                    if move_count % 100 == 0:
                        elapsed_time = time.time() - start_time
                        print(f"Processed {move_count} moves | {game_count} games | Elapsed time: {elapsed_time:.2f}s")

    print(f"Finished processing {move_count} moves from {game_count} games.")

# File paths
input_pgn = "data.pgn"
output_csv = "chess_data.csv"

# Parse the PGN file and write to CSV
parse_pgn(input_pgn, output_csv)