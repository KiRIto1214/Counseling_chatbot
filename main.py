import random
import json
import numpy as np
import pickle

import nltk
import torch
import torch.nn as nn
import torch_utils
# from torch.utlis.data import Dataset ,DataLoader
from nltk.stem import WordNetLemmatizer
from numpy import float32
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from torch import float32
from torch.utils.data import DataLoader, Dataset
from model import NeuralNet

# used to find root words
lemmatizer = WordNetLemmatizer()


# breakdown of sentence into words
def tokanize(sen):
    return nltk.word_tokenize(sen)


def lemma(word):
    return lemmatizer.lemmatize(word.lower())


# collection of the words in array form

# this function return array of zeros and ones , if words in tokenize_sentence founds in collection of all words,
# replace zero with one

# eg if token_sen = [hey, how, are , you] , all_words = [hi, hello , hey ,how, why  are ,is , you , me , i]
# this function will return bag_words = [0 , 0 , 1 , 1 , 0 , 1 , 0 , 1 ,0 , 0]
# this array will be used to train neural network and identify tag of given sentence

def bag_of_word(tokenize_sen, all_words):
    tokenize_sentence = [lemma(w) for w in tokenize_sen]
    bag_word = np.zeros(len(all_words))

    for i, w in enumerate(all_words):
        if w in tokenize_sentence:
            bag_word[i] = 1
    return bag_word


intents = json.loads(open('data_bot.json').read())

words = []
tags = []
doc = []
# letters to ignore
ignore_letter = ['?', '.', ',', '!']


for intent in intents['intent']:
    tag = intent['tag']
    tags.append(tag)
    # collection of tags
    for pattern in intent['patterns']:
        # array of words of patterns in particular tag
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # tokenizing and converting into array
        doc.append((word_list, tag))
        if intent['tag'] not in tags:
            tags.append(intent['tag'])

# array of all words
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letter]
# set for unique words
words = sorted(set(words))

tags = sorted(set(tags))

X_train = []
Y_train = []


# we will train model using pattern sentences and try to predict their tag
for (patrn_sen, tag) in doc:
    bag = bag_of_word(patrn_sen, words)
    X_train.append(bag)
    # here we are using indexes of tags to train [ otherwise we have to encode the tag]
    label = tags.index(tag)
    Y_train.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


input_size = len(X_train[0])
output_size = len(tags)
hidden_size1 = 128
hidden_size2 = 128
batch_size = 32
learning_rate = 0.001
num_epoch = 500

# using torch with above parameters [ 4 layers ANN ]
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
# Checking Cuda , if available then we can use GPU to train ANN
# I have installed for Tensorflow but didn't work for torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size1, hidden_size2, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# stochastic gradient decent optimizer

for epoch in range(num_epoch):
    for (word, label) in train_loader:
        # importing data into device
        word = word.to(device)
        label = label.to(device)

        outputs = model(word.float())
        var = label.clone().detach()
        y_tensor = torch.tensor(var, dtype=torch.long, device=device)

        loss = criterion(outputs, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch + 1}/{num_epoch}, loss={loss.item() :.4f}')

print(f'final loss, loss= {loss.item():.4f}')

# saving model
data = {"model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size1": hidden_size1,
        "hidden_size2": hidden_size2,
        "output_size": output_size,
        "words": words,
        "tags": tags}

FILE = "data.pth"
torch.save(data, FILE)

print("model Saved")

# Similarly, we could have used tensorflow
'''
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for word in words:
        if word in word_patterns:
            bag.append(1)
        else:
            bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = Sequential()

model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(np.array(train_x), np.array(train_y), epochs=400, batch_size=10, verbose=1)
'''
