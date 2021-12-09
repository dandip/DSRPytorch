###############################################################################
# General Information
###############################################################################
# Contains code for converting sampled sequences to Pytorch expressions

###############################################################################
# Dependencies
###############################################################################

from collections import Counter

import torch
import torch.nn as nn

###############################################################################
# Converting Sequence to Operators
###############################################################################

class OperatorNode:
    def __init__(self, operator, arity, parent=None):
        """Description here
        """
        self.operator = operator
        self.arity = arity
        self.parent = parent
        self.left_child = None
        self.right_child = None

    def add_child(self, node):
        if (self.left_child is None):
            self.left_child = node
        elif (self.right_child is None):
            self.right_child = node
        else:
            raise RuntimeError("Both children have been created.")

    def set_parent(self, node):
        self.parent = node

    def remaining_children(self):
        if (self.arity == 0):
            return False
        elif (self.arity == 1 and self.left_child is not None):
            return False
        elif (self.arity == 2 and self.left_child is not None and self.right_child is not None):
            return False
        return True

    def __str__(self):
        return str(self.operator)

    def recursive_print(self, constant_counter=[-1]):
        if (self.arity == 2):
            left_print = self.left_child.recursive_print(constant_counter=constant_counter)
            right_print = self.right_child.recursive_print(constant_counter=constant_counter)
            if (self.left_child.arity == 2):
                left_print = '(' + left_print + ')'
            if (self.right_child.arity == 2):
                right_print = '(' + right_print + ')'
            return str(f"""{left_print} {self.operator} {right_print}""")
        elif (self.arity == 1):
            return str(f"""{self.operator}({self.left_child.recursive_print()})""")
        else:
            if (self.operator == 'c'):
                constant_counter[0] += 1
                return str('c') + str(constant_counter[0])
            return str(self.operator.strip('var_'))

    def torch_print(self, operators, constant_counter=[-1]):
        if (self.arity == 2):
            left_print = self.left_child.torch_print(operators, constant_counter=constant_counter)
            right_print = self.right_child.torch_print(operators, constant_counter=constant_counter)
            return str(f"""{operators.func(self.operator)}({left_print}, {right_print})""")
        elif (self.arity == 1):
            return str(f"""{operators.func(self.operator)}({self.left_child.torch_print(operators, constant_counter=constant_counter)})""")
        else:
            if (self.operator == 'c'):
                constant_counter[0] += 1
                return str('self.c' + str(constant_counter))
            return str('x[:,' + str(operators.var(self.operator)) + ']')

def construct_tree(operators, sequence):
    root = OperatorNode(sequence[0], operators.arity(sequence[0]))
    past_node = root
    for operator in sequence[1:]:
        # Pull next node; this node is the child of the node stored in past_node
        curr_node = OperatorNode(operator, operators.arity(operator), parent=past_node)
        past_node.add_child(curr_node)
        past_node = curr_node
        while (past_node.remaining_children() == False):
            past_node = past_node.parent
            if (past_node is None):
                break
    return root

###############################################################################
# Converting Operators to Pytorch Expression
###############################################################################

class Expression(nn.Module):
    def __init__(self, operators, sequence):
        super(Expression, self).__init__()
        # Generate tree and store root
        self.root = construct_tree(operators, sequence)
        self.num_constants = Counter(sequence)['c']
        if (self.num_constants > 0):
            self.c = torch.nn.Parameter(torch.rand(self.num_constants), requires_grad=True)
        self.expression = self.root.torch_print(operators, constant_counter=[-1])
        self.pretty_expression = self.root.recursive_print(constant_counter=[-1])

    def forward(self, x):
        out = eval(self.expression)
        return out

    def get_constants(self):
        return self.c

    def __str__(self):
        c_expression = self.pretty_expression
        constant_dict = {"c" + str(i): str(float(self.c[i])) for i in range(self.num_constants)}
        for holder, learned_val in constant_dict.items():
            c_expression = c_expression.replace(holder, learned_val)
        return c_expression
