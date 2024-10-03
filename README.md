
# My Story Generator

## Overview

The Story Generator is a Python application that uses deep learning to generate stories based on user-provided text. It utilizes a Long Short-Term Memory (LSTM) model to learn from the input text and generate coherent and contextually relevant sentences. The application supports model training on large text files in chunks, ensuring efficient processing and memory management.

## Features

- **Chunked Text Processing:** Reads large text files in manageable chunks.
- **Tokenizer Support:** Automatically creates and saves a tokenizer for text processing.
- **LSTM Model:** Builds and trains a neural network for story generation.
- **Model Persistence:** Saves trained models and tokenizers to disk for future use.
- **Text Generation:** Generates stories based on a seed text input.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Pickle

You can install the necessary libraries using pip:

```bash
pip install tensorflow numpy
```

## Directory Structure

- `models/` - Directory to store trained models.
- `tokenizers/` - Directory to store tokenizer files.

## Usage

### Initialize the Story Generator

```python
from story_generator import StoryGenerator

story_gen = StoryGenerator()
```

### Training the Model

To train the model on a text file, use the `train_model` method:

```python
story_gen.train_model(user_id='user_1', text_file_path='path/to/textfile.txt')
```

### Generating a Story

After training, you can generate a story with a seed text:

```python
generated_story = story_gen.generate_story(user_id='user_1', seed_text='Once upon a time')
print(generated_story)
```

### Loading a Model and Tokenizer

If you need to load a previously trained model or tokenizer:

```python
model = story_gen.load_model(user_id='user_1')
tokenizer = story_gen.load_tokenizer(user_id='user_1')
```

## Methods

- **`__init__(self, sequence_length=40, max_vocab_size=10000)`**: Initializes the StoryGenerator instance.
- **`load_text_in_chunks(self, file_path, chunk_size=1024 * 1024)`**: Reads a file in chunks.
- **`preprocess_text(self, text, user_id)`**: Preprocesses the input text for model training.
- **`build_model(self, total_words)`**: Constructs the LSTM model.
- **`train_model(self, user_id, text_file_path, epochs=30, batch_size=32)`**: Trains the model on text data.
- **`save_model(self, model, user_id)`**: Saves the trained model to disk.
- **`load_model(self, user_id)`**: Loads a previously trained model.
- **`save_tokenizer(self, tokenizer, user_id)`**: Saves the tokenizer to disk.
- **`load_tokenizer(self, user_id)`**: Loads a previously saved tokenizer.
- **`generate_story(self, user_id, seed_text, max_words=100)`**: Generates a story based on a seed text.

## Logging

TensorFlow logging is set to display only error messages to avoid cluttering the output.



