import pandas as pd
import numpy as np
import chess
import tensorflow as tf
from tensorflow.keras import models # type: ignore
from sklearn.metrics import accuracy_score
import joblib

class ChessEvaluator:
    def __init__(self, model_path, encoder_path):
        self.model = models.load_model(model_path)
        self.move_encoder = joblib.load(encoder_path)
        self.piece_to_value = {
            'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
            'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6
        }
    
    def fen_to_vector(self, fen):
        board = chess.Board(fen)
        vector = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            vector.append(self.piece_to_value.get(piece.symbol(), 0) if piece else 0)
        return np.array(vector, dtype=np.int8)
    
    def predict_move(self, fen):
        board_vector = self.fen_to_vector(fen).reshape(1, -1)
        pred = self.model.predict(board_vector, verbose=0)
        move_idx = np.argmax(pred)
        return self.move_encoder.inverse_transform([move_idx])[0]
    
    def evaluate(self, test_df):
        y_true = test_df["move"].values
        y_pred = [self.predict_move(fen) for fen in test_df["fen"]]
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy

def main():
    # Load test data (use your validation set)
    test_df = pd.read_csv("chess_data.csv").sample(1000)  # Test on 1000 positions
    
    # Initialize evaluators for both models
    best_model = ChessEvaluator("best_model.h5", "move_encoder.pkl")
    final_model = ChessEvaluator("chess_model.h5", "move_encoder.pkl")
    
    # Evaluate both models
    print("Evaluating models...")
    best_acc = best_model.evaluate(test_df)
    final_acc = final_model.evaluate(test_df)
    
    print(f"\n{'Model':<15} {'Accuracy':<10}")
    print("-" * 25)
    print(f"{'Best Model':<15} {best_acc:.4f}")
    print(f"{'Final Model':<15} {final_acc:.4f}")
    
    # Interactive testing
    print("\nInteractive testing (type 'quit' to exit)")
    while True:
        fen = input("\nEnter FEN position: ").strip()
        if fen.lower() == 'quit':
            break
        
        print("\nPredictions:")
        print(f"{'Best Model':<15}: {best_model.predict_move(fen)}")
        print(f"{'Final Model':<15}: {final_model.predict_move(fen)}")

if __name__ == "__main__":
    main()