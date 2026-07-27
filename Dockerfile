# dockerfile for the english to french translator flask app
# using python 3.10 slim to keep the image small

FROM python:3.10-slim

# set the working directory inside the container
WORKDIR /app

# copy just requirements first so docker can cache the pip install layer
COPY requirements.txt .

# install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# now copy the rest of the project (app.py, model/, templates/, static/)
COPY . .

# the flask app listens on port 7860 inside the container
EXPOSE 7860

# start the app with gunicorn (a proper production server, not flask's dev server)
# 1 worker keeps memory usage low, timeout is generous because loading the model takes a few seconds
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "180", "app:app"]
