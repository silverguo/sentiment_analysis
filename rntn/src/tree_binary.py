import numpy as np
import math

__all__ = ['BinaryTree']

# binary tree class
class BinaryTree(object):
    # constructor
    def __init__(self, sentence, structure, label=None):
        self.sentence = sentence
        self.structure = structure
        # number of word
        numWord = len(sentence)
        # initiate node object
        self.nodes = [Node() for idx in range(numWord*2-1)]
        self.leaves= []

        # initiate leaves
        for idx, word in enumerate(sentence):
            node = self.nodes[idx]
            node.word = word
            node.isLeaf = True
            # order of leaf is word index
            node.order = idx
            self.leaves.append(node)

        # tree traverse 
        travs = dict()
        for idx in range(len(self.nodes)):
            node = self.nodes[idx]
            struct = self.structure[idx]
            # struct number is related to parent node
            node.parent = struct - 1
            # add itself to list of children
            self.nodes[node.parent].children.append(idx)
            # get the list for parent node, default empty list
            nList = travs.get(node.parent, [])
            nList.append(idx)
            # save your childre in order
            travs[node.parent] = nList
        # pop fake parent of root
        travs.pop(-1)
        # sorted by key
        self.traverse = sorted(travs.items())

        # give the prob labe
        if label is not None:
            for node in self.leaves:
                # fake left and right child
                node.leftChild = -1
                node.rightChild = -1
                # set prob label to 1
                idxLabel = min(math.floor(label[node.word] / 0.2), 4)
                node.sentiLabel = idxLabel
                node.yreal[idxLabel] = 1.0

            # parent and tuple of children
            for p, [a, b] in self.traverse:
                pNode = self.nodes[p]
                aNode = self.nodes[a]
                bNode = self.nodes[b]
                # word of parent node
                if aNode.order < bNode.order:
                    # add left and right child
                    pNode.leftChild = a
                    pNode.rightChild = b
                    pNode.word = ' '.join([aNode.word, bNode.word])
                else:
                    pNode.leftChild = b
                    pNode.rightChild = a
                    pNode.word = ' '.join([bNode.word, aNode.word])
                # set prob label to 1
                idxLabel = min(math.floor(label[pNode.word] / 0.2), 4)
                pNode.sentiLabel = idxLabel
                pNode.yreal[idxLabel] = 1.0
                # parent order is no matter which child
                pNode.order = aNode.order


class Node(object):
    def __init__(self, word=None, label=None):
        self.word = word
        self.isLeaf = False
        self.order = None
        self.parent = None
        self.children = []
        self.leftChild = None
        self.rightChild = None
        self.yreal = np.zeros(5)
        self.sentiLabel = None



