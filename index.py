from flask import Flask, render_template, request, redirect, url_for, abort
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision
# from torchvision import datasets, transforms
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer
from transformers import BertTokenizer
from datetime import date, datetime, timedelta

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']


# For Data preprocessing
class TransformerTokenizer(torch.nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
    
    def forward(self, input):
        if isinstance(input, list):
            tokens = []
            for text in input:
                tokens.append(self.tokenizer.tokenize(text))
            return tokens
        elif isinstance(input, str):
            return self.tokenizer.tokenize(input)
        raise ValueError(f"Type {type(input)} is not supported.")
        
tokenizer_vocab = vocab(tokenizer.vocab, min_freq=0)


import torchtext.transforms as T

text_transform = T.Sequential(
    TransformerTokenizer(tokenizer),  # Tokenize
    T.VocabTransform(tokenizer_vocab),  # Conver to vocab IDs
    T.Truncate(max_input_length - 2),  # Cut to max length
    T.AddToken(token=tokenizer_vocab["[CLS]"], begin=True),  # BOS token
    T.AddToken(token=tokenizer_vocab["[SEP]"], begin=False),  # EOS token
    T.ToTensor(padding_value=tokenizer_vocab["[PAD]"]),  # Convert to tensor and pad
)


preds_dict_r={'neutral': 0,
 'admiration': 1,
 'approval': 2,
 'gratitude_pride_relief': 3,
 'anger_annoyance_disgust': 4,
 'amusement_excitement_joy': 5,
 'love': 7,
 'confusion_curiosity': 6,
 'disapproval': 8,
 'caring_desire_optimism': 9,
 'disappointment_embarrassment_grief_remorse_sadness': 10,
 'realization': 11,
 'surprise': 12,
 'fear_nervousness': 13}

preds_dict = {v: k for k, v in preds_dict_r.items()}


# Model Definition
from transformers import BertTokenizer, BertModel


try:
    bert = BertModel.from_pretrained('bert-base-uncased')
except Exception as e:
    print(f'Exception! {e}')


class BERTGRUSentiment(nn.Module):
    def __init__(self, bert, hidden_dim, output_dim, n_layers, bidirectional, dropout):        
        super().__init__()
        
        self.bert = bert
        self.embedding_dim = bert.config.to_dict()['hidden_size']
        
        # TODO - Define a GRU layer with n_layers layers
        # bidirectionality conditional on the bidirectional variable, and
        # dropout if there are more than two layers present.
        # Note that the batch dimension should be first.
        # You can take a look at Lab 6 for inspiration on PyTorch's recurrent unit API,
        # or look at the GRU documentation:
        # https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
        self.rnn = nn.GRU(self.embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,#The batch_first parameter is set to True, which means that the input and output tensors will have the batch dimension as the first dimension.
                          dropout = 0 if n_layers < 2 else dropout)
        
        # TODO - Define a linear layer that takes the GRU output and transforms it to a dimensionality
        # of output_dim.
        # Hint: consider what the in_features argument should be if the GRU is bidirectional and each
        # direction has dimensionality of hidden_dim
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        # TODO - Define a dropout layer
        # of the GRU layer's hidden states during training, which helps to prevent the model from 
        #overfitting to the training data and improves its ability to generalize to new data.
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):

        with torch.no_grad():
            embedded = self.bert(text)[0]
        
        _, hidden = self.rnn(embedded)
                
        if self.rnn.bidirectional:
            #In PyTorch, negative indexing can be used to index a tensor from the end, 
            #with -1 referring to the last element along a given dimension, -2 referring 
            #to the second last element, and so on. In this case, hidden[-2,:,:] refers 
            #to the output of the second last layer of the GRU.
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])

        return self.out(hidden)

HIDDEN_DIM = 64  # 254 is better, less than 64 is no very favourable.
OUTPUT_DIM = 14  # We only need one neuron as output
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

model = BERTGRUSentiment(bert, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
model = model.to(DEVICE)

model.load_state_dict(torch.load('/tmp/modelfiles/tut4-model_cpu.pt'))
model = model.eval()

app = Flask(__name__)



def LogActivity(input_time, user_input, prediction_time, predict_result):
    '''For logging the user inputs and running time into daily log file'''
    current = datetime.now()
    fname = 'action_log' + current.strftime('%Y%m%d')
    with open(fname, 'a') as fp:
        fp.write(f"'{input_time}','{user_input}','{prediction_time}','{predict_result}'\n")
    


@app.route("/", methods=["GET", "POST"])
def predict_emotion():
    if request.method == "GET":
        return render_template("index.html")
    if request.method == "POST":
        input_text = request.form["input_text"]

        print(f'***input_text {input_text}')
        
        # DataPreprocessing
        data=text_transform([input_text]).to(DEVICE)
        input_time = datetime.now()
        
        # prediction
        output = model(data)
        prediction = torch.max(output, 1)[1].item()
        result = preds_dict[prediction]
        prediction_time = datetime.now()
        
        LogActivity(input_time, input_text, prediction_time, result)

        return render_template("index.html", display_text=input_text, result=result)


@app.route('/getlog')
def getlog():
    '''Get last 7 days log'''
    
    current = datetime.now()
    
    htmltext = '<p>input_run_time,input_text,end_run_time,result</p>'
    for i in range(0, 8):
        logdate = current - timedelta(days=i)
        fname = 'action_log' + logdate.strftime('%Y%m%d')
        
        try: 
            f1 = open(fname, 'r')
            Lines = f1.readlines()
            f1.close()
            
            htmltext += ''.join(['<p>' + line.strip() + '</p>' for line in Lines])
        except Exception as e:
            continue
        
    return htmltext


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
    # app.run(host='127.0.0.1', debug=True)