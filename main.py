import json
import numpy as np
import tensorflow as tf
from flask import Flask, request

# ================INIT FLASK========================
app = Flask(__name__)


@app.route("/")
def hello_world():
  return "Hello, World!"
# ===================================================


# ================ENDPOINT===========================
@app.route("/predict", methods=["POST"])
def predict():
  return "Predicting..."
# ===================================================


# ================MAIN===============================
if __name__ == "__main__":
  app.run(host='127.0.0.1', port=5000)
# ===================================================
