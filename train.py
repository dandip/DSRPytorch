###############################################################################
# General Information
###############################################################################
# author: dan dipietro | daniel.m.dipietro.22@dartmouth.edu
# description:
# - pytorch implementation of deep symbolic regression (petersen et al.)
# - currently implemented for x^3 + x^2 + x test dataset

###############################################################################
# Dependencies
###############################################################################

import time
import random
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from operators import Operators
from rnn import DSRRNN
from training_utils import *
from expression_utils import *
from collections import Counter

###############################################################################
# Main Training loop
###############################################################################

def main():
    # Hyperparameters
    lr = 0.0005 # Learning rate for sequence-generating RNN
    inner_lr = 0.005 # Learning rate for training of constants within expressions
    entropy_coefficient = 0.005
    risk_factor = 0.60 # We discard the bottom risk_factor quantile of expressions when training the RNN
    batch_size = 500
    num_batches = 200
    hidden_size = 500
    use_gpu = False # (Leave set to False for now)

    # Establish GPU device if necessary
    if (use_gpu and torch.cuda.is_available()):
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Load training and test data
    X_train, X_test, y_train, y_test = get_data()
    X_train, X_test = X_train.to(device), X_test.to(device)
    y_train, y_test = y_train.to(device), y_test.to(device)

    # Initialize operators: all variable operators must have the prefix "var_"
    operator_list = ['*', '+', '-', '/', '^', 'cos', 'sin', 'var_x']
    operators = Operators(operator_list, device)

    # Initialize RNN and optimizer
    dsr_rnn = DSRRNN(operators, hidden_size).to(device)
    optimizer = torch.optim.RMSprop(dsr_rnn.parameters(), lr=lr)

    # Best expression and its performance
    best_expression, best_performance = None, float('-inf')

    # Number of times we've seen each equation
    equation_count = Counter()

    # Outer loop: trains RNN after batches of expressions are generated and benchmarked
    for i in range(num_batches):
        # Inner loop: generates expressions from the RNN and benchmarks them
        sequences = []
        expressions = []
        rewards = []
        log_probabilities = []
        entropies = []
        for j in range(batch_size):
            # Sample sequences and obtain the log probability of generating a given sequence
            seq, log_prob, entropy = sample_sequence(dsr_rnn, operators, device)
            sequences.append(seq)
            log_probabilities.append(log_prob)
            entropies.append(entropy)

        # Update equation counter
        equation_count.update([tuple(x) for x in sequences])

        # Convert sequences into Pytorch expressions that can be evaluated
        for sequence in sequences:
            expressions.append(Expression(operators, sequence).to(device))

        # Optimize constants of expressions (training data)
        for expression in expressions:
            optimize_constants(expression, X_train, y_train, inner_lr)

        # Benchmark expressions (test dataset)
        for expression in expressions:
            rewards.append(benchmark(expression, X_test, y_test))

        # Update best expression
        if (max(rewards) > best_performance):
            best_performance = max(rewards)
            best_expression = expressions[np.argmax(rewards)]

        # Early stopping criteria
        if (best_performance >= 0.98):
            print("~ Early Stopping Met ~")
            print(f"""Best Expression: {best_expression}""")
            break

        # Compute risk threshold
        threshold = np.quantile(rewards, risk_factor)
        indices_to_keep = [i for i in range(len(rewards)) if rewards[i] > threshold]

        # Select corresponding subset of sequences
        sequences = np.array(sequences, dtype='object')[indices_to_keep]
        expressions = np.array(expressions)[indices_to_keep]
        rewards = np.array(rewards)[indices_to_keep]
        log_probabilities = [log_probabilities[j] for j in range(len(log_probabilities)) if j in indices_to_keep]
        entropies = [entropies[j] for j in range(len(entropies)) if j in indices_to_keep]

        # Iterate over subset of sequences above threshold to train RNN
        risk_seeking_grad = 0
        entropy_grad = 0
        for j in range(len(sequences)):
            # Increment risk seeking policy gradient
            risk_seeking_grad += (rewards[j] - threshold) * log_probabilities[j]
            # Increment entropy gradient
            entropy_grad += entropies[j]

        # Mean reduction and clip to limit exploding gradients
        risk_seeking_grad = torch.clip(risk_seeking_grad / len(sequences), -1e6, 1e6)
        entropy_grad = entropy_coefficient * torch.clip(entropy_grad / len(sequences), -1e6, 1e6)

        # Compute loss and backpropagate
        loss = lr * (risk_seeking_grad + entropy_grad)
        loss.backward()
        optimizer.step()

        # Epoch Summary
        print(f"""Epoch: {i} ({int(time.time())})
        Entropy Loss: {entropy_grad.item()}
        Risk-Seeking Loss: {risk_seeking_grad.item()}
        Total Loss: {loss.item()}
        Best Performance (Overall): {best_performance}
        Best Performance (Epoch): {max(rewards)}
        Best Expression (Overall): {best_expression}
        Best Expression (Epoch): {expressions[np.argmax(rewards)]}
        """)

###############################################################################
# Getting Data
###############################################################################

def get_data():
    """Constructs data for model (currently x^3 + x^2 + x)
    """
    X = np.arange(-2, 2.1, 0.1)
    y = X**3 + X**2 + X
    X = X.reshape(X.shape[0], 1)

    # Split randomly
    comb = list(zip(X, y))
    random.shuffle(comb)
    X, y = zip(*comb)

    # Proportion used to train constants versus benchmarking functions
    training_proportion = 0.2
    div = int(training_proportion*len(X))
    X_train, X_test = np.array(X[:div]), np.array(X[div:])
    y_train, y_test = np.array(y[:div]), np.array(y[div:])
    X_train, X_test = torch.Tensor(X_train), torch.Tensor(X_test)
    y_train, y_test = torch.Tensor(y_train), torch.Tensor(y_test)
    return X_train, X_test, y_train, y_test

###############################################################################
# Optimizing constants
###############################################################################

def optimize_constants(expression, X_train, y_train, inner_lr, epochs=500):
    """Optimizes constants of passed expression
    """
    if (expression.num_constants == 0):
        return False
    optim = torch.optim.RMSprop(expression.parameters(), lr=inner_lr)
    loss = torch.nn.L1Loss()
    for i in range(epochs):
        optim.zero_grad()
        y = expression(Variable(X_train, requires_grad=True))
        loss(y, y_train).backward()
        optim.step()

###############################################################################
# Reward function
###############################################################################

def benchmark(expression, X_test, y_test):
    """Obtain reward for a given expression using the passed X_test and y_test
    """
    with torch.no_grad():
        y_pred = expression(X_test)
        return reward_nrmse(y_pred, y_test)

def reward_l1(y_pred, y_test):
    """Compute L1 between predicted y and actual y
    """
    loss = nn.L1Loss()
    val = loss(y_pred, y_test)
    val = min(torch.nan_to_num(val, nan=1e10), torch.tensor(1e10)) # Fix nan and clip
    val = 1 / (1 + val) # Squash
    return val.item()

def reward_nrmse(y_pred, y_test):
    """Compute NRMSE between predicted y and actual y
    """
    loss = nn.MSELoss()
    val = torch.sqrt(loss(y_pred, y_test)) # Convert to RMSE
    val = torch.std(y_test) * val # Normalize using stdev of targets
    val = min(torch.nan_to_num(val, nan=1e10), torch.tensor(1e10)) # Fix nan and clip
    val = 1 / (1 + val) # Squash
    return val.item()

if __name__=='__main__':
    main()
