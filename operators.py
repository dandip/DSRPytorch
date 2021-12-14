###############################################################################
# General Information
###############################################################################
# Author: Daniel DiPietro | dandipietro.com | https://github.com/dandip

# Original Paper: https://arxiv.org/abs/1912.04871 (Petersen et al)

# operators.py: Operator class, which is used as the building blocks for
# assembling PyTorch expressions with an RNN.

###############################################################################
# Dependencies
###############################################################################

import torch

###############################################################################
# Operators Class
###############################################################################

class Operators:
    """
    The list of valid nonvariable operators may be found in nonvar_operators.
    All variable operators must have prefix 'var_'. Constant value operators
    are fine too (e.g. 3.14), but they must be passed as floats.
    """
    nonvar_operators = [
        '*', '+', '-', '/', '^',
        'cos', 'sin', 'tan',
        'exp', 'ln',
        'sqrt', 'square',
        'c' # ephemeral constant
    ]
    nonvar_arity = {
        '*': 2,
        '+': 2,
        '-': 2,
        '/': 2,
        '^': 2,
        'cos': 1,
        'sin': 1,
        'tan': 1,
        'exp': 1,
        'ln': 1,
        'sqrt': 1,
        'square': 1,
        'c': 0
    }
    function_mapping = {
        '*': 'torch.mul',
        '+': 'torch.add',
        '-': 'torch.subtract',
        '/': 'torch.divide',
        '^': 'torch.pow',
        'cos': 'torch.cos',
        'sin': 'torch.sin',
        'tan': 'torch.tan',
        'exp': 'torch.exp',
        'ln': 'torch.log',
        'sqrt': 'torch.sqrt',
        'square': 'torch.square'
    }

    def __init__(self, operator_list, device):
        """Description here
        """
        self.operator_list = operator_list
        self.constant_operators = [x for x in operator_list if x.replace('.', '').strip('-').isnumeric()]
        self.nonvar_operators = [x for x in self.operator_list if "var_" not in x and x not in self.constant_operators]
        self.var_operators = [x for x in operator_list if x not in self.nonvar_operators and x not in self.constant_operators]
        self.__check_operator_list() # Sanity check

        self.device = device

        # Construct data structures for handling arity
        self.arity_dict = dict(self.nonvar_arity, **{x: 0 for x in self.var_operators}, **{x: 0 for x in self.constant_operators})
        self.zero_arity_mask = torch.tensor([1 if self.arity_dict[x]==0 else 0 for x in self.operator_list]).to(device)
        self.nonzero_arity_mask = torch.tensor([1 if self.arity_dict[x]!=0 else 0 for x in self.operator_list]).to(device)
        self.variable_mask = torch.Tensor([1 if x in self.var_operators else 0 for x in self.operator_list])
        self.nonvariable_mask = torch.Tensor([0 if x in self.var_operators else 1 for x in self.operator_list])

        # Contains indices of all operators with arity 2
        self.arity_two = torch.Tensor([i for i in range(len(self.operator_list)) if self.arity_dict[self.operator_list[i]]==2])
        # Contains indices of all operators with arity 1
        self.arity_one = torch.Tensor([i for i in range(len(self.operator_list)) if self.arity_dict[self.operator_list[i]]==1])
        # Contains indices of all operators with arity 0
        self.arity_zero = torch.Tensor([i for i in range(len(self.operator_list)) if self.arity_dict[self.operator_list[i]]==0])
        # Contains indices of all operators that are variables
        self.variable_tensor = torch.Tensor([i for i in range(len(self.operator_list)) if operator_list[i] in self.var_operators])

        # Construct data structures for handling function and variable mappings
        self.func_dict = dict(self.function_mapping)
        self.var_dict = {var: i for i, var in enumerate(self.var_operators)}

    def __check_operator_list(self):
        """Throws exception if operator list is bad
        """
        invalid = [x for x in self.nonvar_operators if x not in Operators.nonvar_operators]
        if (len(invalid) > 0):
            raise ValueError(f"""Invalid operators: {str(invalid)}""")
        return True

    def __getitem__(self, i):
        try:
            return self.operator_list[i]
        except:
            return self.operator_list.index(i)

    def arity(self, operator):
        try:
            return self.arity_dict[operator]
        except NameError:
            print("Invalid operator")

    def arity_i(self, index):
        try:
            return self.arity_dict[self.operator_list[index]]
        except NameError:
            print("Invalid index")

    def func(self, operator):
        return self.func_dict[operator]

    def func_i(self, index):
        return self.func_dict[self.operator_list[index]]

    def var(self, operator):
        return self.var_dict[operator]

    def var_i(self, index):
        return self.var_dict[self.operator_list[index]]

    def __len__(self):
        return len(self.operator_list)
