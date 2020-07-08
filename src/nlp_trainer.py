#!/usr/bin/env python
# coding: utf-8

# # python dependency
# - pytorch
# - torchtext
# 
# # data dependency
# - glove 6b
# - aclImdb

import os
import time
import random
import collections
import tarfile
import torch
from torch import nn
import torchtext.vocab as Vocab
import torch.utils.data as Data
from tqdm import tqdm

# settings
kDevice = "cpu"
kDataDir = "data" # put in the current directory

# read data
def readImdb(data_dir, part_folder):
    text_data_list = []
    # pos and neg are sub folders and show the label info
    for label in ["pos", "neg"]:
        folder_path = os.path.join(data_dir, "aclImdb", part_folder, label)
        for file in tqdm(os.listdir(folder_path)):
            with open(os.path.join(folder_path, file), "rb") as f:
                movie_review = f.read().decode("utf-8").replace('\n', '').lower()
                text_data_list.append([movie_review, 1 if label == "pos" else 0])
    random.shuffle(text_data_list)
    return text_data_list

train_data, test_data = readImdb(kDataDir, "train"), readImdb(kDataDir, "test")

# pre process data
def tokenizer(text):
    return [tok.lower() for tok in text.split(' ')]

def getTokenizedImdb(data):
    # data: list of [string, int]
    return [tokenizer(review) for review, _ in data]

def getImdbVocab(data):
    tokenized_data = getTokenizedImdb(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return Vocab.Vocab(counter, min_freq=5) # filter out the words count less than 5

vocab = getImdbVocab(train_data)

def pad(x, max_len):
    return x[:max_len] if len(x) > max_len else x + [0] * (max_len - len(x))

def preprocessImdb(data, vocab):
    max_len = 500 # pading to 500 words for each review
    tokenized_data = getTokenizedImdb(data)
    features = torch.tensor([pad([vocab.stoi[word] for word in words], max_len) for words in tokenized_data])
    labels = torch.tensor([score for _, score in data])
    return features, labels

batch_size = 64
train_set = Data.TensorDataset(*preprocessImdb(train_data, vocab))
test_set = Data.TensorDataset(*preprocessImdb(test_data, vocab))
train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
test_iter = Data.DataLoader(test_set, batch_size)

# model
class TextRNN(nn.Module):
    def __init__(self, vocab_len, embed_size, num_hiddens, num_layers):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_len, embed_size)
        
        # bidrectional lstm
        self.encoder = nn.LSTM(input_size=embed_size,
                              hidden_size=num_hiddens,
                              num_layers=num_layers,
                              bidirectional=True)
        # full connect layer
        self.decoder = nn.Linear(4 * num_hiddens, 2)
        
    def forward(self, inputs):
        # inputs shape: (batch_size, words_len)
        # inverse inputs and fetch the attributes, outputs shape: (words_len, batch_size, word_vec_dim)
        embeddings = self.embedding(inputs.permute(1, 0))
        outputs, _ = self.encoder(embeddings)
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        outs = self.decoder(encoding)
        return outs

# build a 2 hidden layer bidirectional nural network
embed_size, num_hiddens, num_layers = 100, 100, 2
net = TextRNN(len(vocab), embed_size, num_hiddens, num_layers) # make sure the model args are convienient for C++

# download and cache larget scale pretrained vocab from torchtext
# website link: https://nlp.stanford.edu/projects/glove
# domestic link: https://sunyanhust.github.io/post/nlp-chang-yong-mo-xing-he-shu-ju-ji-gao-su-xia-zai/
# you can manually down load the glove.6B.100d.zip, rename as glove.6B.zip and put in the cache dir
glove_vocab = Vocab.GloVe(name='6B', dim=100, cache=os.path.join(kDataDir, "glove"))

def loadPretrainedEmbedding(words, pretrained_vocab):
    embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0])
    oov_count = 0 # out of vocabulary
    for i, word in enumerate(words):
        try:
            idx = pretrained_vocab.stoi[word]
            embed[i, :] = pretrained_vocab.vectors[idx]
        except KeyError:
            oov_count += 1 # ?
    
    if oov_count > 0:
        print ("there are %d oov words" % oov_count)
        
    return embed

net.embedding.weight.data.copy_(loadPretrainedEmbedding(vocab.itos, glove_vocab))
net.embedding.weight.requires_grad = False # pretrained data no need to udpate

# train
def evaluate_accuracy(data_iter, net, device=None):
    if device is None:
        # if not specified device, use net device
        device = list(net.parameters())[0].device 
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval() # eval mode will close dropout
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train() # back to train mode
            n += y.shape[0]
    return acc_sum / n

def train(net, train_iter, test_iter, batch_size, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
            
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

lr = 0.01
num_epochs = 5

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
loss = nn.CrossEntropyLoss()
train(net, train_iter, test_iter, batch_size, loss, optimizer, kDevice, num_epochs) # the training may take a long time in cpu

# predict
def predict(net, vocab, sentence):
    device = list(net.parameters())[0].device
    words = tokenizer(sentence)
    sentence_tensor = torch.tensor([vocab.stoi[word] for word in words], device=device)
    output = net(sentence_tensor.view((1, -1)))
    label = torch.argmax(output, dim=1)
    print ("output:", output)
    print ("label:", label.item())
    return "positive" if label.item() == 1 else "negative"

sentence1 = "I feel the movie kind of great and to my taste"
sentence_tensor1 = torch.tensor([vocab.stoi[word] for word in tokenizer(sentence1)], device=list(net.parameters())[0].device).view(1, -1) # display the input tensor for C++ use
print ("input:", sentence_tensor1)

res = predict(net, vocab, sentence1)
print (res)

sentence2 = "the movie has bad experience"
sentence_tensor2 = torch.tensor([vocab.stoi[word] for word in tokenizer(sentence2)], device=list(net.parameters())[0].device).view(1, -1) # display the input tensor for C++ use
print ("input:", sentence_tensor2)

res = predict(net, vocab, sentence2)
print (res)

# export model
example_sentence = "funny movie and make me exciting"
example_sentence_tensor = torch.tensor([vocab.stoi[word] for word in tokenizer(sentence2)], device=list(net.parameters())[0].device).view(1, -1)
traced_script_module = torch.jit.trace(net, example_sentence_tensor)
traced_script_module.save("text_rnn.pt")

# use the exported model to predict
predict(traced_script_module, vocab, sentence2)

