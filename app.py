# Flask app for the English-to-French translator

from flask import Flask, render_template, request
import os
import pickle

import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Embedding,
    Input,
    LSTM,
    RepeatVector,
    TimeDistributed,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
HF_REPO_ID = "Fatima173/english-french-translator-model"
HF_MODEL_FILENAME = "model.weights.h5"

# Load the tokenizers and sequence lengths saved from the notebook.
with open(os.path.join(MODEL_DIR, "en_tokenizer.pkl"), "rb") as file:
    en_tokenizer = pickle.load(file)

with open(os.path.join(MODEL_DIR, "fr_tokenizer.pkl"), "rb") as file:
    fr_tokenizer = pickle.load(file)

with open(os.path.join(MODEL_DIR, "config.pkl"), "rb") as file:
    config = pickle.load(file)

max_en_len = config["max_en_len"]
max_fr_len = config["max_fr_len"]
en_vocab_size = len(en_tokenizer.word_index) + 1
fr_vocab_size = len(fr_tokenizer.word_index) + 1


def build_translation_model():
    """Rebuild the same model architecture used in the notebook."""
    translation_model = Sequential([
        Input(shape=(max_en_len,), dtype="int32"),
        Embedding(input_dim=en_vocab_size, output_dim=128),
        Bidirectional(LSTM(256)),
        RepeatVector(max_fr_len),
        LSTM(512, return_sequences=True),
        TimeDistributed(Dense(fr_vocab_size, activation="softmax")),
    ])
    return translation_model


print("Downloading model weights from Hugging Face Hub...")
weights_path = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename=HF_MODEL_FILENAME,
)

model = build_translation_model()
model.load_weights(weights_path)
print("Model weights loaded successfully.")


def translate(sentence):
    sentence = sentence.lower().strip()
    if not sentence:
        return ""

    tokens = en_tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(tokens, maxlen=max_en_len, padding="post")

    prediction = model.predict(padded, verbose=0)
    predicted_ids = np.argmax(prediction[0], axis=-1)

    words = [
        fr_tokenizer.index_word.get(int(word_id), "")
        for word_id in predicted_ids
        if word_id > 0
    ]
    return " ".join(word for word in words if word)


@app.route("/", methods=["GET", "POST"])
def home():
    english_input = ""
    french_output = ""
    error_message = ""

    if request.method == "POST":
        english_input = request.form.get("english_text", "").strip()
        if english_input:
            try:
                french_output = translate(english_input)
                if not french_output:
                    error_message = "The model could not translate that sentence."
            except Exception as error:
                print(f"Translation error: {error}")
                error_message = "Something went wrong while translating."

    return render_template(
        "index.html",
        english_input=english_input,
        french_output=french_output,
        error_message=error_message,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)

