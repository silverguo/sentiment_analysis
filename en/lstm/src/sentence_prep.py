 import pickle

__all__ = ['Sentence_Prep']

class Sentence_Prep(object):
    # sentence preprocessing

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

    # list of tokens to idx
    def sentence_encode(self, token_list):
        pass

    
