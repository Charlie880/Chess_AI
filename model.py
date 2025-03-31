import pandas as pd
import numpy as np
import chess
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.model_selection import train_test_split

class ChessDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size, move_encoder):
        self.df = df
        self.batch_size = batch_size
        self.move_encoder = move_encoder
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_df = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X = np.array([self.fen_to_vector(fen) for fen in batch_df["fen"]])
        y = self.move_encoder.transform(batch_df["move"])
        return X, y

    def on_epoch_end(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    def fen_to_vector(self, fen):
        piece_to_value = {
            'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
            'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6
        }
        board = chess.Board(fen)
        vector = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            vector.append(piece_to_value.get(piece.symbol(), 0) if piece else 0)
        return np.array(vector, dtype=np.int8)

def main():
    # Load dataset
    df = pd.read_csv("chess_data.csv")
    print(f"Loaded {len(df)} positions")
    
    # Prepare move encoder
    try:
        move_encoder = joblib.load("move_encoder.pkl")
        print("Loaded existing move encoder")
    except FileNotFoundError:
        print("Creating new move encoder")
        move_encoder = LabelEncoder()
        move_encoder.fit(df["move"])
        joblib.dump(move_encoder, "move_encoder.pkl")
    
    # Create data generator
    batch_size = 128
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_gen = ChessDataGenerator(train_df, batch_size, move_encoder)
    val_gen = ChessDataGenerator(val_df, batch_size, move_encoder)
    
    # Build model
    model = models.Sequential([
        layers.Input(shape=(64,)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(move_encoder.classes_), activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=30,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True)
        ]
    )
    
    # Save final model
    model.save("chess_model.h5")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()