from src import *
import pickle

if __name__ == '__main__':
    # load dict
    with open('./data/temp/imdb_dict.pkl', 'rb') as f:
        dictReview = pickle.load(f)

    # demo
    print(dictReview['train']['pos'][0])
    
