import numpy as np
from os import path
import pandas as pd
from .tree_binary import BinaryTree

__all__ = ['loadStree', 'loadSostr', 'loadDsplit', 
           'loadDict', 'loadSentlabel', 'dataPrep']

def loadStree(filePath, fileName='STree.txt'):
    with open(path.join(filePath, fileName), 'r') as f:
        structTree = [np.array(line.split('|')).astype(int) 
                      for line in f.readlines()]
    return structTree

def loadSostr(filePath, fileName='SOStr.txt'):
    sentences = []
    lexicon = set([])
    with open(path.join(filePath, fileName), 'r') as f:
        for line in f.readlines():
            words = line.strip().split('|')
            sentences.append(words)
            lexicon = lexicon.union(words)
    return sentences, lexicon

def loadDsplit(filePath, fileName='datasetSplit.txt'):
    return pd.read_csv(path.join(filePath, fileName), 
                       sep=',')['splitset_label'] \
             .tolist()

def loadDict(filePath, fileName='dictionary.txt'):
    with open(path.join(filePath, fileName), 'r') as f:
        idxDict = dict()
        for line in f.readlines():
            coup = line.split('|')
            idxDict[int(coup[1])] = coup[0]
    return idxDict

def loadSentlabel(filePath, fileName='sentiment_labels.txt'):
    return pd.read_csv(path.join(filePath, fileName), 
                       sep='|') \
             .set_index('phrase ids')['sentiment values'] \
             .to_dict()

def dataPrep(filePath):
    structTree = loadStree(filePath)
    sentences, lexicon = loadSostr(filePath)
    sampleSplit = loadDsplit(filePath)
    idxDict = loadDict(filePath)
    sentLabel = loadSentlabel(filePath)
    
    dictLabel = dict()
    for key, value in idxDict.items():
        dictLabel[value] = sentLabel[key]

    allTree = dict()
    allTree[1] = []
    allTree[2] = []
    allTree[3] = []
    for idx in range(len(structTree)):
        sentence = sentences[idx]
        structure = structTree[idx]
        dSplit = sampleSplit[idx]
        allTree[dSplit].append(BinaryTree(sentence, 
                                          structure, 
                                          dictLabel))
    return lexicon, allTree

