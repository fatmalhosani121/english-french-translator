# main flask app for the english to french translator
# the trained keras model is hosted on Hugging Face Hub
# and downloaded when the app starts

from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os
from huggingface_hub import hf_hub_download

# create the flask app
app = Flask(__name__)

# path to the folder that holds the small files (tokenizers + config)
MODEL_DIR = "model"

# hugging face repo details where the trained model file lives
HF_REPO_ID = "Fatima173/english-french-translator-model"
HF_MODEL_FILENAME = "model.h5"

# download the model file from hugging face hub
print("Downloading model from Hugging Face Hub...")
model_path = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename=HF_MODEL_FILENAME
)
print(f"Model file ready at {model_path}")

# load the keras model into memory
model = tf.keras.models.load_model(model_path)
print("Keras model loaded.")

# load the tokenizers from the local model folder
with open(os.path.join(MODEL_DIR, "en_tokenizer.pkl"), "rb") as f:
    en_tokenizer = pickle.load(f)

with open(os.path.join(MODEL_DIR, "fr_tokenizer.pkl"), "rb") as f:
    fr_tokenizer = pickle.load(f)

# load the max lengths saved during training
with open(os.path.join(MODEL_DIR, "config.pkl"), "rb") as f:
    config = pickle.load(f)

max_en_len = config["max_en_len"]
print("Tokenizers and config loaded.")


# helper function
def translate(sentence):
    sentence = sentence.lower().strip()

    if not sentence:
        return ""

    # convert English sentence to sequence
    tokens = en_tokenizer.texts_to_sequences([sentence])

    # pad to the required length
    padded = pad_sequences(tokens, maxlen=max_en_len, padding="post")

    # predict French sequence
    prediction = model.predict(padded, verbose=0)
    predicted_ids = np.argmax(prediction[0], axis=-1)

    # convert IDs back to words
    index_to_word = fr_tokenizer.index_word
    words = [
        index_to_word.get(idx, "")
        for idx in predicted_ids
        if idx > 0
    ]

    return " ".join(words)


@app.route("/", methods=["GET", "POST"])
def home():
    english_input = ""
    french_output = ""

    if request.method == "POST":
        english_input = request.form.get("english_text", "")
        french_output = translate(english_input)

    return render_template(
        "index.html",
        english_input=english_input,
        french_output=french_output,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
