# English to French Translator

This project is an English to French translation web application developed using TensorFlow, Flask, Docker, Hugging Face, and Render.

The application allows users to enter a short English sentence and receive a French translation using a trained sequence-to-sequence LSTM model.

## Features

- English to French translation
- Simple web interface using Flask
- Dockerized application
- Online deployment using Render
- Model hosted on Hugging Face

## Technologies

- Python
- TensorFlow / Keras
- Flask
- Docker
- Hugging Face Hub
- Render

## Project Structure

```
app.py
requirements.txt
Dockerfile
model/
templates/
static/
```

## Live Demo

Render Deployment:

https://english-french-translator-4voh.onrender.com

## How to Run

1. Clone the repository.

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Run the application.

```bash
python app.py
```

4. Open your browser:

```
http://localhost:7860
```

## Limitations

The model was trained on a relatively small dataset, so some translations may not be grammatically perfect.

## Author

Fatima Al Hosani
