# DSRPytorch
Pytorch implementation of *[Deep Symbolic Regression](https://arxiv.org/pdf/1912.04871.pdf)* (Petersen et al.)

## File Tree
* *train.py*: Contains main training loop
* *rnn.py*: Contains PyTorch RNN used to sample tokens
* *training_utils.py*: Contains code for constructing sequences with the RNN and applying in-situ constraints
* *expression_utils.py*: Converts sequences to PyTorch expressions that can be evaluated and have their constants optimized.
* *operators.py*: Contains logic for operators (arity, corresponding PyTorch functions, etc.)
