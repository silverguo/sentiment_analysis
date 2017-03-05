import glob
import os
from collections import defaultdict

__all__ = ['ImdbLoader']

class ImdbLoader:
    # imdb dataset loader

    # constructor
    def __init__(self, imdbPath):
        self.imdbPath = imdbPath
    
    # directory scan
    def dir_scan(self, dir):
        listPos = glob.glob(os.path.join(dir, 'pos') + '/*')
        listNeg = glob.glob(os.path.join(dir, 'pos') + '/*')
        return listPos, listNeg

    # load review
    def review_load(self, listPos):
        listReview = []
        for l in listPos:
            with open(l, 'r') as f:
                listReview.append(f.readline())
        return listReview


    # load dict
    def dict_load(self):
        dictReview = defaultdict(defaultdict)
        # train data
        listPos, listNeg = self.dir_scan(os.path.join(self.imdbPath, 'train'))
        # read review
        dictReview['train']['pos'] = self.review_load(listPos)
        dictReview['train']['neg'] = self.review_load(listNeg)
        # test data
        listPos, listNeg = self.dir_scan(os.path.join(self.imdbPath, 'test'))
        # read review
        dictReview['test']['pos'] = self.review_load(listPos)
        dictReview['test']['neg'] = self.review_load(listNeg)

        return dictReview

        
