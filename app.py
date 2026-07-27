# main flask app for the english to french translator
# loads the trained LSTM model once on startup and translates
# whatever english sentence the user types on the webpage

from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os

# create the flask app
app = Flask(__name__)

# path to the model folder
MODEL_DIR = "model"

# load the saved model
# this happens once when the app starts so predictions are fast later
print("Loading model...")
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "model.keras"))

# load the tokenizers
with open(os.path.join(MODEL_DIR, "en_tokenizer.pkl"), "rb") as f:
    en_tokenizer = pickle.load(f)

with open(os.path.join(MODEL_DIR, "fr_tokenizer.pkl"), "rb") as f:
    fr_tokenizer = pickle.load(f)

# load the max lengths we saved from training
with open(os.path.join(MODEL_DIR, "config.pkl"), "rb") as f:
    config = pickle.load(f)

max_en_len = config["max_en_len"]
print("Model and tokenizers loaded.")


# helper function - same idea as the one i used in the colab notebook
def translate(sentence):
    sentence = sentence.lower().strip()
    if not sentence:
        return ""

    # turn the english sentence into numbers with the tokenizer
    tokens = en_tokenizer.texts_to_sequences([sentence])

    # pad to the same length as during training
    padded = pad_sequences(tokens, maxlen=max_en_len, padding="post")

    # ask the model to predict the french sequence
    prediction = model.predict(padded, verbose=0)
    predicted_ids = np.argmax(prediction[0], axis=-1)

    # convert numbers back into french words
    index_to_word = fr_tokenizer.index_word
    words = [index_to_word.get(idx, "") for idx in predicted_ids if idx > 0]

    return " ".join(words)


# main page - shows the form
@app.route("/", methods=["GET", "POST"])
def home():
    english_input = ""
    french_output = ""

    # if the user submitted the form, run the translation
    if request.method == "POST":
        english_input = request.form.get("english_text", "")
        french_output = translate(english_input)

    return render_template(
        "index.html",
        english_input=english_input,
        french_output=french_output,
    )


# start the flask server
# host=0.0.0.0 makes it work inside docker
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
