from src import *
import pickle

# preprocess for imdb
def imdb_prep():
    # load data
    dl = ImdbLoader('./data/aclImdb')
    dictReview = dl.dict_load()
    """
    dictReview
    - train
        - sentence
        - label
    - test
        - sentence
        - label
    """
    numSample = len(dictReview['train']['sentence']) + \
                len(dictReview['test']['sentence'])
    print('IMDB dataset, {} sentences are loaded'.format(numSample))

    # preprocessing
    en_prep = Text_Prep(lang='en')
    w2v_prep = Embedding_Prep(w2v_pickle='./data/w2v_dl/en_dl_500000.pickle')

    imdb_input = dict()
    imdb_input['X_train'] = [w2v_prep.token_encode(en_prep.tokenizer(t), 
                                                   vocab_size=200000) 
                             for t in dictReview['train']['sentence']]
    imdb_input['y_train'] = dictReview['train']['label']
    imdb_input['X_test'] = [w2v_prep.token_encode(en_prep.tokenizer(t), 
                                                  vocab_size=200000) 
                            for t in dictReview['test']['sentence']]
    imdb_input['y_test'] = dictReview['test']['label']

    # serializable
    with open('./data/model_input/imdb_input.pickle', 'wb') as f:
        pickle.dump(imdb_input, f)
    

# main
if __name__ == '__main__':
    imdb_prep()

