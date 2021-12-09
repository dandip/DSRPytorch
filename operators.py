###############################################################################
# General Information
###############################################################################
# Logic for operators (arity, corresponding PyTorch functions, allowed
# operators, etc)

###############################################################################
# Dependencies
###############################################################################

import torch

###############################################################################
# Operators Class
###############################################################################

class Operators:
    nonvar_operators = [
        '*', '+', '-', '/', '^',
        'cos', 'sin',
        'exp', 'ln',
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
        'exp': 1,
        'ln': 1,
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
        'exp': 'torch.exp',
        'ln': 'torch.log',
    }

    def __init__(self, operator_list, device):
        """Description here
        """
        self.operator_list = operator_list
        self.nonvar_operators = [x for x in self.operator_list if "var_" not in x]
        self.var_operators = [x for x in operator_list if x not in self.nonvar_operators]
        self.__check_operator_list() # Sanity check

        self.device = device

        # Construct data structures for handling arity
        self.arity_dict = dict(self.nonvar_arity, **{x: 0 for x in self.var_operators})
        self.zero_arity_mask = torch.tensor([1 if self.arity_dict[x]==0 else 0 for x in self.operator_list]).to(device)
        self.nonzero_arity_mask = torch.tensor([1 if self.arity_dict[x]!=0 else 0 for x in self.operator_list]).to(device)

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

    def func(self, operator):
        return self.func_dict[operator]

    def var(self, operator):
        return self.var_dict[operator]


    def __len__(self):
        return len(self.operator_list)
