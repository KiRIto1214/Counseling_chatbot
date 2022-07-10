import random
import json
import numpy as np
import pickle
from model import NeuralNet
import nltk
import torch
from main import bag_of_word
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FILE = "data.pth"
intents = json.loads(open('data_bot.json').read())
data = torch.load(FILE)

# importing saved model
input_size = data["input_size"]
hidden_size1 = data["hidden_size1"]
hidden_size2 = data["hidden_size2"]
output_size = data["output_size"]
words = data["words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size1, hidden_size2, output_size).to(device)

model.load_state_dict(model_state)
model.eval()

print("Bot is online")

while 1:
    user_input = input("You - ")

    if user_input == "quit":
        break

    # tokenize the sen
    user_input = nltk.word_tokenize(user_input)
    # return array
    x = bag_of_word(user_input, words)

    x = x.reshape(1, x.shape[0])
    # numpy array
    x = torch.from_numpy(x).to(device)
    # predict labels / index of tags
    output = model(x.float())
    _, predicted = torch.max(output, dim=1)
    idx = predicted.detach().clone()
    tag = tags[idx.item()]

    prob = torch.softmax(output, dim=1)
    # probability
    pro = prob[0][predicted.item()]

    # check by probability
    if pro.item() > 0.60:
        for intent in intents["intent"]:
            if tag == intent["tag"]:
                # Random output
                print("Maple:", random.choice(intent["responses"]))
    else:
        print("I don't Understand, please Rephrase it")