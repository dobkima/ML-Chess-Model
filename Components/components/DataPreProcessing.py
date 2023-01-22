import os
import chess
import chess.pgn
import numpy as np
import pandas as pd
from mldesigner import command_component, Input, Output


# Converts a chess board to a matrix representation
#  1 = white pawn,  2 = white knight,  3 = white bishop,  4 = white rook,  5 = white queen,  6 = white king
# -1 = black pawn, -2 = black knight, -3 = black bishop, -4 = black rook, -5 = black queen, -6 = black king
#  0 = empty square
def matrix_representation(board):
    matrix_representation = np.zeros((8,8), dtype=int)
    for i in range(8):
        for j in range(8):
            piece = board.piece_at(chess.square(j,i))
            if piece is not None:
                if piece.color == chess.WHITE:
                    matrix_representation[i,j] = piece.piece_type
                else:
                    matrix_representation[i,j] = -piece.piece_type
    return matrix_representation

# Reads in the games from the pgn file and returns a dictionary with the games split into rating ranges of 100
def read_in_games(pgn_file):
    rating_bins = {}
    with open(pgn_file) as pgn:
        
        while True:
            try:
                game = chess.pgn.read_game(pgn)
                if game.headers["TimeControl"] == "600+0":
                    white_elo = int(game.headers["WhiteElo"])
                    black_elo = int(game.headers["BlackElo"])
                        
                    white_range = str(white_elo // 100)
                    black_range = str(black_elo // 100)
                        
                    if white_range == black_range:
                        if white_range in rating_bins:
                            rating_bins[white_range].append(game)
                        else:
                            rating_bins[white_range] = [game]
            except:
                break

        return rating_bins

def split_game_into_positions(rating_bins):

    positions = {}

    for rating_range in rating_bins:
        positions[rating_range] = []

        for game in rating_bins[rating_range]:
            board = game.board()
            for move in game.mainline_moves():
                position = matrix_representation(board)
                start_square, end_square = move.from_square, move.to_square
                turn = 1 if board.turn == chess.WHITE else -1
                positions[rating_range].append((position, turn, start_square, end_square))
                board.push(move)

    # remove ranges with less than 1000000 positions
    for rating_range in list(positions.keys()):
        if len(positions[rating_range]) < 1000000:
            del positions[rating_range]

    return positions
            
def convert_positions_to_csv(positions, csv_file):
    df = pd.DataFrame(columns=range(68), dtype=int)
    df.index.names = ["pos"]
    df.rename(columns={64: "rating", 65: "turn", 66: "start", 67: "end"}, inplace=True)
    for i in range(8):
        for j in range(8):
            df.rename(columns={i*8+j: chess.square_name(chess.square(j,i))}, inplace=True)

    for rating_range in positions:        
        for position in positions[rating_range]:
            rating = int(rating_range)
            board = position[0].flatten()
            turn = position[1]
            start = position[2]
            end = position[3]
            df.loc[len(df)] = np.append(board, [rating, turn, start, end])
    
    df.drop_duplicates(inplace=True)
    df.to_csv(csv_file, index=False)


# Define the pipeline component
@command_component(
    name="game_data_pre_processing",
    version="1.0.0",
    display_name="Pre-Process Game Data",
    description="Cleans and converts the data (Chess Positions) to a standard format.",
    environment="conda.yaml"
)
def data_preprocessing(
    pgn_files: Input(type="uri_folder", description="PGN file containing chess games"), 
    csv_file: Output(type="uri_file", description="Folder containing the prepared data"),
    file_name = "data.pgn"
    ):
    games_path = os.path.join(pgn_files, file_name)
    rating_bins = read_in_games(games_path)
    positions = split_game_into_positions(rating_bins)
    convert_positions_to_csv(positions, csv_file)

