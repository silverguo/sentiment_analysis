import pickle
import spacy

__all__ = ['Embedding_Prep', 'Text_Prep']

class Embedding_Prep(object):
    # embedding preprocessing

    # constructor
    def __init__(self, w2v_pickle):
        with open(w2v_pickle, 'rb') as f:
            self.word_embedding = pickle.load(f)
        """
        word_embedding
        - embedding
        - vocab
            - idx_word
            - word_idx
        """
        self.word_idx = self.word_embedding['vocab']['word_idx']


    # list of tokens to idx
    def token_encode(self, token_list, vocab_size=200000):
        idx_list = []
        for t in token_list:
            i = self.word_idx.get(t, -1)
            # if exceed the vocab size
            if i == -1 or i >= vocab_size:
                i = vocab_size - 1
            idx_list.append(i)
        return idx_list
    
class Text_Prep(object):
    # document preprocessing

    # constructor
    def __init__(self, lang='en'):
        self.nlp = spacy.load(lang)
    
    # tokenizer
    def tokenizer(self, text):
        doc = self.nlp(text)
        return [w.text.lower() for w in doc]

    
