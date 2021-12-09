###############################################################################
# General Information
###############################################################################
# RNN used to sample operators

###############################################################################
# Dependencies
###############################################################################

import torch.nn as nn
import torch

###############################################################################
# RNN Class
###############################################################################

class DSRRNN(nn.Module):
    def __init__(self, operators, hidden_size):
        super(DSRRNN, self).__init__()

        self.input_size = 2*len(operators) # One-hot encoded parent and sibling
        self.hidden_size = hidden_size
        self.output_size = len(operators) # Output is a softmax distribution over all operators
        self.operators = operators

        self.i2h = nn.Linear(
            self.input_size + self.hidden_size, self.hidden_size
        )
        self.i2o = nn.Linear(
            self.input_size + self.hidden_size, self.output_size
        )
        self.activation = nn.Softmax(dim=1)

    def forward(self, input, hidden):
        """Input should be [parent, sibling]
        """
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.activation(self.i2o(combined))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

    def init_input(self):
        return torch.zeros(1, self.input_size)
