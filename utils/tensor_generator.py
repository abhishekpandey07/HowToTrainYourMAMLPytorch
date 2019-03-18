class TensorGenerator(object):
    """
        Class to keep returning the same tensor until the end of time
    """
    def __init__(self,tensor,max_length):
        self.tensor = tensor
        self.length = max_length
        
    def __len__(self):
        return self.length

    def __iter__(self):        
        return self

    def __next__(self):
        return self.tensor

        
        
