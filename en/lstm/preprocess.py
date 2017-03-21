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

    # input dict
    imdb_input = dict()
    # prepare train, valid
    idx_train, idx_valid = data_split(len(dictReview['train']['label']), 
                                      split_ratio=0.8)

    # tokenize and word encode
    # train and valid
    print('{} train sample'.format(len(idx_train)))
    imdb_input['X_train'] = [w2v_prep.token_encode(en_prep.tokenizer(t), 
                                                   vocab_size=200000, 
                                                   len_max=50) 
                             for t in [dictReview['train']['sentence'][idx] 
                                       for idx in idx_train]]
    imdb_input['y_train'] = [dictReview['train']['label'][idx] for idx in idx_train]

    print('{} valid sample'.format(len(idx_valid)))
    imdb_input['X_valid'] = [w2v_prep.token_encode(en_prep.tokenizer(t), 
                                                   vocab_size=200000, 
                                                   len_max=50)  
                             for t in [dictReview['train']['sentence'][idx] 
                                       for idx in idx_valid]]
    imdb_input['y_valid'] = [dictReview['train']['label'][idx] for idx in idx_valid]
    
    # test
    print('{} test sample'.format(len(dictReview['test']['label'])))
    imdb_input['X_test'] = [w2v_prep.token_encode(en_prep.tokenizer(t), 
                                                  vocab_size=200000, 
                                                  len_max=50) 
                            for t in dictReview['test']['sentence']]
    imdb_input['y_test'] = dictReview['test']['label']

    # serializable
    with open('./demo/data/imdb_input.pickle', 'wb') as f:
        pickle.dump(imdb_input, f)
    

# main
if __name__ == '__main__':
    imdb_prep()

