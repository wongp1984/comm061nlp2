from flask import Flask, render_template, request, redirect, url_for, abort
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision
# from torchvision import datasets, transforms
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer
from transformers import BertTokenizer
from datetime import date, datetime


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def predict_emotion():
    if request.method == "GET":
        return render_template("index.html")
    
if __name__ == "__main__":
    app.run(debug=True)