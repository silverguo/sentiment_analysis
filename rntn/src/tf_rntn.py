import tensorflow as tf
import os
from .data_load import dataPrep


__all__ = ['RecursiveNTensorN']

# rntn model class
class RecursiveNTensorN():

    def __init__(self):
        self.tfSeed = 666
        self.uWeight = 0.0001
        self.wordSize = 10
        self.labelNum = 5
        self.coefL2 = 0.02
        self.learnRate = 0.001
        self.iterNum = 1000
        self.modelDir = './tf_checkpoint/'
        self.modelReload = 'model_last'

    # load data, train, test and dev
    def load_data(self, filePath):
        self.lexicon, self.allTree = dataPrep(filePath)
        self.vocab = dict()
        idxWord = 0
        for word in self.lexicon:
            self.vocab[word] = idxWord
            idxWord += 1

    # input placeholders
    def add_input_placeholder(self):
        self.isLeafPh = tf.placeholder(tf.bool, (None), name='isLeafPh')
        self.leftChildPh = tf.placeholder(tf.int32, (None), name='leftChildPh')
        self.rightChildPh = tf.placeholder(tf.int32, (None), name='rightChildPh')
        self.nodeIndicePh = tf.placeholder(tf.int32, (None), name='nodeIndicePh')
        self.labelPh = tf.placeholder(tf.int32, (None), name='labelPh')
 
    # initiate parameters of rntn
    def add_model_variable(self):
        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable(name='wVector', 
                                             shape=[len(self.lexicon), 
                                                    self.wordSize])
        with tf.variable_scope('weights'):
            self.tensorV = tf.get_variable(name='tensorV', 
                                           shape=[2 * self.wordSize, 
                                                  2 * self.wordSize, 
                                                  self.wordSize])
            self.linearW = tf.get_variable(name='linearW', 
                                           shape=[2 * self.wordSize, 
                                                  self.wordSize])
            self.softW = tf.get_variable(name='softW', 
                                         shape=[self.wordSize, 
                                                self.labelNum])
        with tf.variable_scope('bias'):
            self.linearB = tf.get_variable(name='linearB', 
                                           shape=[1, self.wordSize])
            self.softB = tf.get_variable(name='softB', 
                                         shape=[1, self.labelNum])
        self.modelArray = tf.TensorArray(tf.float32, size=0, 
                                         dynamic_size=True, 
                                         clear_after_read=False, 
                                         infer_shape=False)

    # word vector indice
    def word_indice(self, wordIndice):
        return tf.expand_dims(tf.gather(self.embedding, 
                                        wordIndice), 0)

    # children layer
    def children_layer(self, leftTensor, rightTensor):
        return tf.concat(1, [leftTensor, rightTensor])

    # tensor layer
    def tensor_layer(self, childrenTensor):
        return tf.matmul(childrenTensor, 
                         tf.reshape(tf.matmul(childrenTensor,
                                              tf.reshape(self.tensorV, 
                                                         [self.wordSize * 2, 
                                                          self.wordSize * self.wordSize * 2])), 
                                    [self.wordSize * 2, self.wordSize]))
                         
    # weight layer
    def weight_layer(self, childrenTensor):
        return tf.matmul(self.linearW, childrenTensor)

    # hidden layer
    def hidden_layer(self, leftTensor, rightTensor):
        childrenTensor = self.children_layer(leftTensor, rightTensor)
        return tf.nn.relu(self.weight_layer(childrenTensor) + 
                          self.tensor_layer(childrenTensor) + 
                          self.linearB)
                   
    # loop body for while
    def loop_body(self, idx):
        nodeIndice = tf.gather(self.nodeIndicePh, idx)
        leftChild = tf.gather(self.leftChildPh, idx)
        rightChild = tf.gather(self.rightChildPh, idx)
        nodeVector = tf.cond(tf.gather(self.isLeafPh, idx), 
                             lambda: self.word_indice(nodeIndice), 
                             lambda: self.hidden_layer(self.modelArray.read(leftChild), 
                                                       self.modelArray.read(rightChild)))
        self.modelArray = self.modelArray.write(idx, nodeVector)
        idx = tf.add(idx, 1)
        return idx

    # loop condition for while
    def loop_cond(self, idx):
        return tf.less(idx, tf.squeeze(tf.shape(self.isLeafPh)))

    # build the recursive graph
    def build_graph(self):
        # set graph level seed
        tf.set_random_seed(self.tfSeed)
        # input and variable intiate
        self.add_input_placeholder()
        self.add_model_variable()
        # while loop for hidden layer
        self.modelArray, _ = tf.while_loop(self.loop_cond, 
                          self.loop_body, 
                          [0])
        # softmax layer
        self.logit = tf.matmul(self.modelArray.concat(), self.softW) + self.softB
        self.rootLogit = tf.gather(self.logit, self.modelArray.size()-1)
        self.rootPred = tf.squeeze(tf.argmax(self.rootLogit, 1))

        # loss function
        regLoss = self.coefL2 * (tf.nn.l2_loss(self.tensorV) + 
                                 tf.nn.l2_loss(self.linearW) + 
                                 tf.nn.l2_loss(self.softW))
        self.fullLoss = regLoss + \
                        tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(self.logit, 
                                                                                     self.labelPh))
        self.rootLoss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(self.rootLogit, 
                                                                                     self.labelPh[-1]))
        # training optimizer
        self.train_op = tf.train.GradientDescentOptimizer(self.learnRate).minimize(self.fullLoss)
        # self.train_op = tf.train.AdamOptimizer(self.learnRate).minimize(self.fullLoss)

    # input dict feed
    def build_feed_dict(self, sTree):
        feed_dict = {
                     self.isLeafPh: [node.isLeaf
                                     for node in sTree.nodes], 
                     self.leftChildPh: [node.leftChild
                                        for node in sTree.nodes], 
                     self.rightChildPh: [node.rightChild
                                         for node in sTree.nodes], 
                     self.nodeIndicePh: [self.vocab.get(node.word, -1)
                                         for node in sTree.nodes], 
                     self.labelPh: [node.sentiLabel
                                    for node in sTree.nodes]
        }
        return feed_dict

    # training iteration
    def iter_train(self, newModel=False):
        lossHistory = []
        with tf.Session() as sess:
            if newModel:
                sess.run(tf.initialize_all_variables())
            else:
                saver = tf.train.Saver()
                saver.restore(sess, self.modelDir + self.modelReload)
            selIdx = tf.random_uniform([50], minval=0, maxval=len(allTree['1']), 
                                       dtype=tf.int32)
            for idx in selIdx:
                feed_dict = build_feed_dict(allTree['1'][idx])
                idxLoss, _ = sess.run([self.fullLoss, self.train_op], 
                                       feed_dict=feed_dict)
                lossHistory.append(idxLoss)
            saver = tf.train.Saver()
            if not os.path.exists(self.modelDir):
                os.makedirs(self.modelDir)
            saver.save(sess, self.modelDir + self.modelReload)
        # dev error
        return lossHistory

    # training model
    def model_train(self, filePath):
        self.load_data(filePath)
        self.build_graph()

        # train history         
        lossHistory = dict()
        lossHistory['train'] = []
        # lossHistory['dev'] = []
        # accHistory = dict()
        # accHistory['train'] = []
        # accHistory['dev'] = []

        # initial iter
        trainLoss = self.iter_train(newModel=True)
        lossHistory['train'].append(trainLoss)

        # iteration for training
        for iterIdx in range(self.iterNum):
            trainLoss = self.iter_train(newModel=False)
            lossHistory['train'].append(trainLoss)





# https://github.com/bogatyy/cs224d/blob/master/assignment3/rnn_static_graph.py#L56
