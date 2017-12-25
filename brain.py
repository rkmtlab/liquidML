from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from threading import (Event, Thread)
import time
event = Event()

def valid_sample(sample):
    return not sample in ['', '\n']

def split_input_target(line):
    input, target = line.split(DELIM)
    input = list(map(lambda x:float(x), input.split(',')))
    if valid_sample(target):
        target = list(map(lambda x:float(x), target.split(',')))
    return input, target

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        # LSTMCell(input_size, hidden_size)
        self.L = HLS
        self.IN = 2
        self.OUT = 2
        self.lstm1 = nn.LSTMCell(self.IN, self.L)
        self.lstm2 = nn.LSTMCell(self.L, self.OUT)

    def forward(self, input, future = 0):
        # input.data.shape => [19 x 99 x 21]
        # input.size(0) => 19
        # input.size(1) => 99
        outputs = []
        h_t = Variable(torch.zeros(input.size(0), self.L).double(), requires_grad=False)
        c_t = Variable(torch.zeros(input.size(0), self.L).double(), requires_grad=False)
        h_t2 = Variable(torch.zeros(input.size(0), self.OUT).double(), requires_grad=False)
        c_t2 = Variable(torch.zeros(input.size(0), self.OUT).double(), requires_grad=False)
 
        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            # input_t.data.shape => [19 x 1 x 2]
            input_t = torch.squeeze(input_t)
            # input_t.data.shape => [19 x 2]
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            # outputs += [h_t]
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            outputs += [h_t2]
        for i in range(future):
            h_t, c_t = self.lstm1(h_t2, (h_t, c_t))
            # outputs += [h_t]
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            outputs += [h_t2]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs
# build the model
# This could be RNN
# nn.Module (parent class) .double() convert float into double
# Loss function: loss = criterion(nn.output, target)
# use pytorch.optim.LBFGS as optimizer since we can load the whole data to train
# MSE stands for Mean Square Error
TEACHER_FILE = 'teacher.txt'
DELIM = ':'
BACH_SIZE = 10
MIN_CNT_BACH = 10
ITERATION = 100
LR = 1.0 # Learning rate
TL = 0.8 # To Learn: rate to spare for learning 1.0 - TL will be in test
HLS = 100 # Hidden Layer Size in each LSTM gate

seq = Sequence()
seq.double()
criterion = nn.MSELoss()
optimizer = optim.LBFGS(seq.parameters(), lr=LR)
np.random.seed(0)
torch.manual_seed(0)
def valid_line(line):
    return DELIM in line

def load(file_name):
    input_rst, target_rst = [], []
    input_bach, target_bach = [], []
    with open(file_name) as f:
        lines = f.read().split('\n')
        for line in lines:
            if not valid_line(line):
                continue
            input, target = split_input_target(line)
            input_bach.append(input)
            target_bach.append(target)
            if len(input_bach) >= BACH_SIZE:
                input_rst.append(input_bach)
                target_rst.append(target_bach)
                input_bach = []
                target_bach = []
    input_rst = np.array(input_rst)
    target_rst = np.array(target_rst)
    return input_rst, target_rst

def list2variable(a):
    return Variable(torch.from_numpy(a), requires_grad=False)

# Candidate to remove
def plot_line(line, c_in='r', m_in='x'):
    for p in line:
        plt.scatter(p[0], p[1], c=c_in, marker=m_in)

def saveplt(input, pred, future, name):
    for line in input.data:
        plot_line(line, c_in='b')
    for line in pred.data:
        past = line[:-future]
        post = line[-future:]
        plot_line(past, c_in='g')
        plot_line(post, c_in='r')
    plt.savefig(name)
    plt.close()

def learn():
    while True:
        event.wait()
        input_src, target_src = load(TEACHER_FILE)
        if len(input_src) < MIN_CNT_BACH:
            # print("Samples (about): ", len(input_src) * BACH_SIZE)
            # print("Data not enough. Samples must be >", MIN_CNT_BACH * BACH_SIZE)
            pass
        else:
            LL = int(len(input_src) * TL)
            input = list2variable(input_src[:LL])
            target = list2variable(target_src[:LL])
            test_input = list2variable(input_src[LL:])
            test_target = list2variable(target_src[LL:])
            for i in range(ITERATION):
                print('STEP: ', i)
                def closure():
                    optimizer.zero_grad()
                    out = seq(input)
                    loss = criterion(out, target)
                    print('loss:', loss.data.numpy()[0])
                    loss.backward()
                    return loss
                optimizer.step(closure)
                future = 10
                pred = seq(test_input, future = future)
                loss = criterion(pred[:, :-future], test_target)
                print('test loss:', loss.data.numpy()[0])
                # saveplt(input, pred, future, 'predict%d.pdf' % i)
        print("End of iteration")
        event.clear()

def push_last_line(file_name, s):
    with open(file_name, 'a') as f:
        f.write('\n')
        f.write(s)

def join_input_target(input, target):
    input = ','.join(map(str, input))
    target = ','.join(map(str, target))
    return input + DELIM + target

def answer(input):
    input = np.array([[input]])
    input = list2variable(input)
    rst = seq(input)
    return list(rst.data[0][0])

def learn_and_answer(line):
    if not valid_line(line):
        return "invalid line:" + line
    input, target = split_input_target(line)
    if valid_sample(target):
       push_last_line(TEACHER_FILE, line)
       event.set() # Restart learning_thread
    return ','.join(map(str, answer(input)))

learning_thread = Thread(target=learn)
event.clear()
learning_thread.start()
