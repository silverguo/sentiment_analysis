import glob
import os

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
        print(len(listPos))
        print(len(listNeg))
        return listPos, listNeg

    # load set
    def load_set(self):
        # train data
        self.dir_scan(os.path.join(self.imdbPath, 'train'))
        
