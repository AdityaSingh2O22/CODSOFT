import os
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from PIL import Image, ImageDraw, ImageFont


text_data = "H:\CODESOFT\data\small\boxing\boxing\1.jpg"


# Create a vocabulary mapping each unique character to a unique integer
chars = sorted(set(text_data))
char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}

# Add a special token for unknown characters
UNK_TOKEN = 'UNK'
char_to_int[UNK_TOKEN] = len(chars)
int_to_char[len(chars)] = UNK_TOKEN

# Convert the text data into numerical sequences with handling unknown characters
sequences = [char_to_int.get(char, char_to_int[UNK_TOKEN]) for char in text_data]
sequence_length = 100  # Adjust this value depending on the desired sequence length

# Split the dataset into input sequences and target characters
input_sequences = []
target_characters = []

for i in range(0, len(sequences) - sequence_length, 1):
    input_seq = sequences[i:i + sequence_length]
    target_char = sequences[i + sequence_length]
    input_sequences.append(input_seq)
    target_characters.append(target_char)

# Reshape the input sequences to (samples, sequence_length, features)
num_samples = len(input_sequences)
input_sequences = np.reshape(input_sequences, (num_samples, sequence_length, 1))

# Normalize the input data to values between 0 and 1
input_sequences = input_sequences / float(len(chars))

# Convert the target characters to one-hot encoding
target_characters = tf.keras.utils.to_categorical(target_characters, num_classes=len(chars) + 1)

# Build the RNN model
model = Sequential()
model.add(LSTM(256, input_shape=(input_sequences.shape[1], input_sequences.shape[2])))
model.add(Dense(len(chars) + 1, activation='softmax'))  # +1 for the UNK_TOKEN

model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
epochs = 50  # You can adjust the number of epochs based on the dataset and model performance
batch_size = 128

# Wrap the model.fit() inside a tf.function
@tf.function
def train_step(inputs, targets):
    return model.train_on_batch(inputs, targets)

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for i in range(0, num_samples - batch_size, batch_size):
        batch_inputs = input_sequences[i:i + batch_size]
        batch_targets = target_characters[i:i + batch_size]
        loss = train_step(batch_inputs, batch_targets)
        print(f"Batch {i // batch_size + 1}/{num_samples // batch_size} - Loss: {loss}")

# Function to generate text
def generate_text(model, seed_text, num_chars):
    generated_text = seed_text
    while len(generated_text) < num_chars:
        x = [char_to_int.get(char, char_to_int[UNK_TOKEN]) for char in generated_text[-sequence_length:]]
        x = np.array(x)
        x = np.pad(x, (sequence_length - len(x), 0), 'constant', constant_values=char_to_int[UNK_TOKEN])
        x = np.reshape(x, (1, sequence_length, 1))
        x = x / float(len(chars))
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = int_to_char.get(index, UNK_TOKEN)
        generated_text += result
    return generated_text

# Function to generate text with temperature scaling
def generate_text(model, seed_text, num_chars, temperature=1.0):
    generated_text = seed_text
    while len(generated_text) < num_chars:
        x = [char_to_int.get(char, char_to_int[UNK_TOKEN]) for char in generated_text[-sequence_length:]]
        x = np.array(x)
        x = np.pad(x, (sequence_length - len(x), 0), 'constant', constant_values=char_to_int[UNK_TOKEN])
        x = np.reshape(x, (1, sequence_length, 1))
        x = x / float(len(chars))

        # Use the model to predict the next character probabilities with temperature scaling
        prediction = model.predict(x, verbose=0)[0]
        prediction = np.log(prediction) / temperature
        exp_prediction = np.exp(prediction)
        prediction = exp_prediction / np.sum(exp_prediction)  # Normalize probabilities

        # Sample the next character from the probability distribution
        index = np.random.choice(len(prediction), p=prediction)
        result = int_to_char.get(index, UNK_TOKEN)
        generated_text += result
    return generated_text




# Seed text to start the generation
seed_text = "Your initial seed text here...\n"
num_chars_to_generate = 500

# generated_text = generate_text(model, seed_text, num_chars_to_generate)
# print(generated_text)

# Function to convert text to images
def text_to_images(text):
    font = ImageFont.load_default()  # You can choose a different font and size
    image_width, image_height = font.getsize(text)
    image = Image.new("RGB", (image_width, image_height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, font=font, fill="black")
    return image

# Create a folder to store the generated images
output_folder = "generated_images"
os.makedirs(output_folder, exist_ok=True)

# Generate text using your model
generated_text = generate_text(model, seed_text, num_chars_to_generate)

# Define a function to sanitize a character for file naming
def sanitize_character(char):
    valid_characters = set(string.ascii_letters + string.digits + "_")
    return char if char in valid_characters else "_"

# Sanitize the generated text for file naming
sanitized_generated_text = "".join(sanitize_character(char) for char in generated_text)

# Convert generated text to images
generated_images = []
for char in generated_text:
    char_image = text_to_images(char)
    generated_images.append(char_image)

# Convert generated text to images and save to the specified folder
for idx, char in enumerate(generated_text):
    char_image = text_to_images(char)
    file_name = f"generated_image_{idx}_{char}.png"
    file_path = os.path.join(output_folder, file_name)
    char_image.save(file_path)  # Save image to the specified folder
