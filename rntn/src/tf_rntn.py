import tensorflow as tf
import .data_load import dataPrep


__all__ = ['RecursiveNTensorN']

# rntn model class
class RecursiveNTensorN():

    def __init__(self):
        self.uWeight = 0.0001
        self.wordSize = 10
        self.labelNum = 5
        self.coefL2 = 0.02
        self.learnRate = 0.001

    # load data, train, test and dev
    def load_data(self, filePath):
        self.lexicon, self.allTree = dataPrep(filePath)
        self.vocab = dict()
        idxWord = 0
        for word in lexicon:
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
                                             [len(self.lexicon), 
                                              self.wordSize])
        with tf.variable_scope('weights'):
            self.tensorV = tf.get_variable(name='tensorV', 
                                           [2 * self.wordSize, 
                                            2 * self.wordSize, 
                                            self.wordSize])
            self.linearW = tf.get_variable(name='linearW', 
                                           [2 * self.wordSize, 
                                            self.wordSize])
            self.softW = tf.get_variable(name='softW', 
                                         [self.wordSize, 
                                          self.labelNum])
        with tf.variable_scope('bias'):
            self.linearB = tf.get_variable(name='linearB', 
                                           [1, self.wordSize])
            self.softB = tf.get_variable(name='softB', 
                                         [1, self.labelNum])
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
                                              tf.reshape(tensorV, 
                                                         [self.wordSize * 2, 
                                                          self.wordSize * self.wordSize * 2])), 
                                    [self.wordSize * 2, self.wordSize]))
                         
    # weight layer
    def weight_layer(self, childrenTensor):
        return tf.matmul(linearW, childrenTensor)

    # hidden layer
    def hidden_layer(self, leftTensor, rightTensor):
        childrenTensor = children_layer(leftTensor, rightTensor)
        return tf.nn.relu(weight_layer(self, childrenTensor) + 
                          tensor_layer(self, childrenTensor) + 
                          linearB)
                   
    # loop body for while
    def loop_body(self, idx):
        nodeIndice = tf.gather(self.nodeIndicePh, idx)
        leftChild = tf.gather(self.leftChildPh, idx)
        rightChild = tf.gather(self.rightChildPh, idx)
        nodeVector = tf.cond(tf.gather(self.isLeafPh, idx), 
                             lambda: word_indice(nodeIndice), 
                             lambda: hidden_layer(self.modelArray.read(leftChild), 
                                                  self.modelArray.read(rightChild)))
        self.modelArray = modelArray.write(idx, nodeVector)
        idx = tf.add(idx, 1)
        return idx

    # loop condition for while
    def loop_cond(self, idx):
        return tf.less(idx, tf.squeeze(tf.shape(self.isLeafPh)))

    # build the recursive graph
    def build_graph(self):
        add_input_placeholder(self)
        add_model_variable(self)
        # while loop for hidden layer
        self.modelArray, _ = tf.while_loop(loop_cond, 
                                           loop_body, 
                                           [self.modelArray, 0], 
                                           parallel_iterations=1)
        # softmax layer
        self.logit = tf.matmul(self.modelArray.concat(), self.softW) + self.softB
        self.rootLogit = tf.gather(self.logit, self.modelArray.size()-1)
        self.rootPred = tf.squeeze(tf.argmax(self.rootLogit, 1))

        # loss function
        regLoss = self.coefL2 * (tf.nn.l2_loss(self.tensorV), 
                                 tf.nn.l2_loss(self.linearW), 
                                 tf.nn.l2_loss(self.softW))
        self.fullLoss = regLoss + 
                        tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(self.logit, 
                                                                                     self.labelPh))
        self.rootLoss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(self.rootLogit, 
                                                                                     self.labelPh[-1]))
        # training optimizer
        self.train_op = tf.train.AdamOptimizer(self.learnRate).minimize(self.fullLoss)

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

    # training model
    def model_train(self, filePath):
        load_data(self, filePath)
        build_graph(self)



# https://github.com/bogatyy/cs224d/blob/master/assignment3/rnn_static_graph.py#L56
