class SimpleTokenizerV1:
    
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
        
    def encode(self, text):
        """
        """
        if type(text) == list:
            return [self.str_to_int[word] for word in text]
        else:
            return self.str_to_int[text]
    
    def decode(self, ids):
        if type(ids) == list:
            return [self.int_to_str[id] for id in ids]
        else:
            return self.int_to_str[ids]
