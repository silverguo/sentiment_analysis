from src import *

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

    # prep for batch training
    
    
    


# main
if __name__ == '__main__':
    imdb_prep()
