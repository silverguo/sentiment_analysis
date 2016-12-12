import tensorflow as tf
import .data_load import dataPrep


__all__ = ['RecursiveNTensorN']

# rntn model class
class RecursiveNTensorN():

    def __init__(self):
        self.uWeight = 0.0001
        self.wordSize = 10
        self.labelNum = 5

    # load data, train, test and dev
    def load_data(self, filePath):
        self.lexicon, self.allTree = dataPrep(filePath)

    # input placeholders
    def add_input_placeholder():
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
                                           [self.wordSize, 
                                            2 * self.wordSize])
            self.softW = tf.get_variable(name='softW', 
                                         [self.labelNum, 
                                          self.wordSize])
        with tf.variable_scope('bias'):
            self.linearB = tf.get_variable(name='linearB', 
                                           [self.wordSize, 1])
            self.softB = tf.get_variable(name='softB', 
                                         [self.labelNum, 1])

    # word vector indice
    def word_indice(self, wordIndice):
        return tf.expand_dims(tf.gather(self.embedding, 
                                        wordIndice), 0)

    # children layer
    def children_layer(self, leftTensor, rightTensor):
        return tf.concat(0, [leftTensor, rightTensor])

    # tensor layer
    def tensor_layer(self, childrenTensor):
        return tf.matmul(tf.transpose(tf.reshape(tf.matmul(tf.transpose(childrenTensor),
                                                           tf.reshape(tensorV, 
                                                                      [self.wordSize * 2, 
                                                                       self.wordSize * self.wordSize * 2])), 
                                                 [self.wordSize * 2, self.wordSize])), 
                         childrenTensor)

    # weight layer
    def weight_layer(self, childrenTensor):
        return tf.matmul(linearW, childrenTensor)

    # hidden layer
    def hidden_layer(self, leftTensor, rightTensor):
        childrenTensor = children_layer(leftTensor, rightTensor)
        return tf.nn.relu(weight_layer(self, childrenTensor) + 
                          tensor_layer(self, childrenTensor) + 
                          linearB)
                   
    # def 


    def inference(self, sentenceTree, )
