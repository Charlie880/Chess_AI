import chess
import tensorflow as tf
import joblib
import numpy as np

class ChessAI:
    def __init__(self, model_path='chess_model.h5', encoder_path='move_encoder.pkl'):
        self.model = tf.keras.models.load_model(model_path)
        self.move_encoder = joblib.load(encoder_path)
        self.piece_map = {
            'P':1, 'N':2, 'B':3, 'R':4, 'Q':5, 'K':6,
            'p':-1, 'n':-2, 'b':-3, 'r':-4, 'q':-5, 'k':-6
        }

    def fen_to_vector(self, fen):
        board = chess.Board(fen)
        return np.array([self.piece_map.get(p.symbol(), 0) 
                        if (p := board.piece_at(sq)) else 0 
                        for sq in chess.SQUARES], dtype=np.int8)

    def predict(self, fen):
        vector = self.fen_to_vector(fen).reshape(1, 64)
        pred = self.model.predict(vector, verbose=0)
        return self.move_encoder.inverse_transform([np.argmax(pred)])[0]

def explain_illegal_move(board, move_uci):
    """Explain why a move is illegal"""
    try:
        move = board.parse_uci(move_uci)
        if move not in board.legal_moves:
            if not any(m.uci() == move_uci for m in board.generate_legal_moves()):
                # Check if path is blocked for sliding pieces
                piece = board.piece_at(move.from_square)
                if piece and piece.piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
                    between = chess.SquareSet.between(move.from_square, move.to_square)
                    blockers = [sq for sq in between if board.piece_at(sq)]
                    if blockers:
                        return f"Path blocked by {', '.join(board.piece_at(sq).symbol() for sq in blockers)}"
                
                # Check if king would be in check
                board.push(move)
                if board.is_check():
                    return "Move would leave king in check"
                board.pop()
                
                return "No legal path exists for this piece"
    except:
        pass
    return "Invalid move pattern for this piece type"

def self_play(max_moves=40):
    ai = ChessAI()
    board = chess.Board()
    move_history = []
    
    print("=== AI Self-Play Debug Mode ===")
    print("Move sequence (with legal move check):")
    
    while not board.is_game_over() and len(move_history) < max_moves:
        current_fen = board.fen()
        ai_move_uci = ai.predict(current_fen)
        
        try:
            move = board.parse_uci(ai_move_uci)
            if move not in board.legal_moves:
                explanation = explain_illegal_move(board, ai_move_uci)
                raise chess.IllegalMoveError(f"{ai_move_uci} - {explanation}")
            
            # Display move
            move_num = len(move_history) // 2 + 1
            prefix = f"{move_num}... " if (len(move_history) % 2 == 1) else f"{move_num}. "
            print(prefix + ai_move_uci)
            
            move_history.append(ai_move_uci)
            board.push(move)
            
        except (chess.IllegalMoveError, chess.InvalidMoveError) as e:
            print(f"\n🚨 Illegal move detected: {e}")
            print("Current position analysis:")
            print(f"FEN: {current_fen}")
            print(f"Turn: {'White' if board.turn else 'Black'}")
            print("Legal moves sample:", [m.uci() for m in list(board.legal_moves)[:5]], "...")
            break
    
    print("\nGame terminated.")
    print(f"Moves played: {len(move_history)}")
    print("Final FEN:", board.fen())
    if board.is_game_over():
        print("Game result:", board.result())

if __name__ == "__main__":
    self_play()