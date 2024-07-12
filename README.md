# -Music-Generation-with-AI
Create an AI-powered music generation system capable  of composing original music. Utilize deep learning  techniques like Recurrent Neural Networks (RNNs) or  Generative Adversarial Networks (GANs) to generate  music sequences.
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

def generate_music(num_notes, sequence_length):
    # Define LSTM model with a unique architecture
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(sequence_length, 2)))
    model.add(LSTM(32))
    model.add(Dense(2))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Generate music sequence using a custom algorithm
    music_sequence = []
    for i in range(num_notes):
        input_sequence = np.random.rand(1, sequence_length, 2)
        output = model.predict(input_sequence)
        music_sequence.append(output[0])

        # Apply a custom transformation to the output
        music_sequence[-1] = np.tanh(music_sequence[-1] * 2) + 0.5

    return music_sequence

# Train the model on a custom dataset
def train_model(dataset):
    # Split dataset into training and testing sets
    train_data, test_data = dataset.split(test_size=0.2, random_state=42)

    # Encode music data using a custom encoding scheme
    train_encoded = []
    for music_piece in train_data:
        encoded_piece = []
        for note in music_piece:
            encoded_note = [note.pitch * 2, note.duration / 2]
            encoded_piece.append(encoded_note)
        train_encoded.append(encoded_piece)

    test_encoded = []
    for music_piece in test_data:
        encoded_piece = []
        for note in music_piece:
            encoded_note = [note.pitch * 2, note.duration / 2]
            encoded_piece.append(encoded_note)
        test_encoded.append(encoded_piece)

    # Train the model
    model.fit(train_encoded, epochs=100, batch_size=32, validation_data=test_encoded)

# Generate music using the trained model
def generate_music_piece(num_notes):
    music_sequence = generate_music(num_notes, sequence_length=10)
    music_piece = []
    for note in music_sequence:
        pitch = int(note[0] * 127)
        duration = int(note[1] * 100)
        music_piece.append((pitch, duration))
    return music_piece

# Save the generated music piece to a file
def save_music_piece(music_piece, filename):
    with open(filename, 'w') as f:
        for note in music_piece:
            f.write(f'{note[0]} {note[1]}\n')

# Example usage
dataset = load_dataset('custom_dataset')
train_model(dataset)
music_piece = generate_music_piece(100)
save_music_piece(music_piece, 'generated_music.txt')

