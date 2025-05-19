# Next Word Prediction using RNN

This project uses a Recurrent Neural Network (LSTM) to predict the next word in a sequence of text.

## Features
- Tokenizes a text corpus into n-grams
- Trains an RNN to learn word sequences
- Uses LSTM and Embedding layers
- Saves the trained model

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Pandas

## Dataset
Expected format: a CSV file named `text_dataset.csv` with a single column `Text`, containing full sentences.

| Text                        |
|-----------------------------|
| The quick brown fox         |
| I am learning machine       |

## Instructions

1. Place your dataset in the project directory as `text_dataset.csv`.
2. Run `next_word_prediction.py` to train the model.
3. The trained model will be saved as `next_word_model.h5`.

## How it Works
- Tokenizes sentences
- Converts them into padded n-gram sequences
- Feeds them to an LSTM model to learn next word prediction

## Output
- Model accuracy during training
- Saved model (`.h5`) file for future inference

## License
MIT License
