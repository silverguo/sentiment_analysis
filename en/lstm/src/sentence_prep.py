 import pickle

__all__ = ['Sentence_Prep']

class Sentence_Prep(object):
    # sentence preprocessing

    # constructor
    def __init__(self, w2v_pickle):
        with open(w2v_pickle, 'rb') as f:
            self.word_embedding = pickle.load(f)
        self.word_idx = self.word_embedding['vocab']['word_idx']
        """
        word_embedding
        - embedding
        - vocab
            - idx_word
            - word_idx
        """


    # list of tokens to idx
    def sentence_encode(self, token_list, vocab_size=200000, 
                        load_embedding=False):
        return [self.word_idx.get(t) for t in token_list]

    
