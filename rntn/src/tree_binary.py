import numpy as np

__all__ = ['BinaryTree', 'Node']

class BinaryTree(object):
    def __init__(self, sentence, structure, label=None):
        self.sentence = sentence
        self.structure = structure
        numWord = len(sentence)
        self.nodes = [Node() for idx in range(numWord*2-1)]
        self.leaves= []

        for idx, word in enumerate(sentence):
            node = self.nodes[idx]
            node.word = word
            node.order = idx
            self.leaves.append(node)

        travs = dict()
        for idx in range(len(self.nodes)):
            node = self.nodes[idx]
            struct = self.structure[idx]
            node.parent = struct - 1
            self.nodes[node.parent].childrens.append(idx)
            nList = travs.get(node.parent, [])
            nList.append(idx)
            travs[node.parent] = nList
        travs.pop(-1)
        self.traverse = sorted(travs.items())

        if label is not None:
            for node in self.leaves:
                node.yreal[0] = label[node.word]
                node.yreal[1] = 1 - node.yreal[0]

            for p, [a, b] in self.traverse:
                pNode = self.nodes[p]
                aNode = self.nodes[a]
                bNode = self.nodes[b]
                if aNode.order < bNode.order:
                    pNode.word = ' '.join([aNode.word, bNode.word])
                else:
                    pNode.word = ' '.join([bNode.word, aNode.word])
                pNode.yreal[0] = label[pNode.word]
                pNode.yreal[1] = 1 - pNode.yreal[0]
                pNode.order = aNode.order


class Node(object):
    def __init__(self, word=None, label=None):
        self.word = word
        self.order = None
        self.parent = None
        self.childrens = []
        self.yreal = np.zeros(2)

        self.ypred = None
        self.X = None
        self.d = None


