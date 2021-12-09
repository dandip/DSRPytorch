###############################################################################
# General Information
###############################################################################
# Contains code for sampling sequences from the RNN

###############################################################################
# Dependencies
###############################################################################

import torch
import torch.nn as nn

###############################################################################
# Sequence sampling
###############################################################################

def sample_sequence(rnn, operators, device, min_length=3, max_length=15):
    """Samples a symbolic sequence from the RNN of specified min/max length
    """
    token_list = []
    cum_log_prob = 0 # Cumulative log probability of the expression
    entropy = 0 # Cumulative entropy of the expression
    counter = 1 # Number of tokens left to sample
    parent, sibling = -1, -1
    input_tensor = rnn.init_input()
    hidden_tensor = rnn.init_hidden()
    eps = torch.Tensor([1e-20 for i in range(len(operators))]).to(device)

    # Sample tokens until all arities are met
    while counter > 0:
        output, hidden_tensor = rnn.forward(input_tensor.to(device), hidden_tensor.to(device))

        # Apply constraints
        output = apply_constraints(output, operators, token_list, counter, min_length, max_length, device, eps)
        output = output / torch.sum(output)

        # Sample from categorical distribution and append token to traversal
        dist = torch.distributions.Categorical(output)
        token = dist.sample()
        log_prob = dist.log_prob(token)

        token = operators[token.item()]
        token_list.append(token)

        # Increment log probability
        cum_log_prob += log_prob

        # Increment entropy
        entropy += dist.entropy()

        # Update counter
        counter += operators.arity(token) - 1

        # Compute next parent and sibling; assemble next input tensor
        if (counter != 0):
            parent, sibling = get_parent_sibling(token_list, operators)
            input_tensor = get_input_tensor(parent, sibling, operators)
    return token_list, cum_log_prob, entropy

def get_parent_sibling(token_list, operators):
    """Returns parent, sibling for the most recent token in token_list
    """
    recent = len(token_list)-1
    if (operators.arity(token_list[recent]) > 0):
        return token_list[recent], -1
    counter = 0
    for i in range(recent, -1, -1):
        counter += operators.arity(token_list[i]) - 1
        if (counter == 0):
            return token_list[i], token_list[i+1]

def get_input_tensor(parent, sibling, operators):
    """Constructs input tensor for RNN based on parent/sibling
    """
    parent_tensor = torch.eye(len(operators))[operators[parent]]
    try:
        sibling_tensor = torch.eye(len(operators))[operators[sibling]]
    except:
        sibling_tensor = torch.zeros(len(operators))
    concat = torch.cat((parent_tensor, sibling_tensor))
    return concat[None, :]

def apply_constraints(output, operators, token_list, counter, min_length, max_length, device, eps):
    """Applies in situ constraints to the distribution contained in output based on the current tokens
    """
    # Add small epsilon to output so that there is a probability of selecting
    # everything (otherwise can get NaN values after applying constraints)
    output = output + eps

    # Check that minimum length is met
    if (len(token_list) + counter < min_length):
        # Only nonzero arity from now on
        output = torch.minimum(output, operators.nonzero_arity_mask)

    # Check that maximum length is met
    if (len(token_list) + counter >= max_length):
        # Only zero arity from now on
        output = torch.minimum(output, operators.zero_arity_mask)
    return output
