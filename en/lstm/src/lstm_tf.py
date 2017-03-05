import tensorflow as tf

__all__ = ['SentiLstm']

# lstm for sentiment analysis
class SentiLstm:

    # constructor
    def __init__(self):
        pass

    # config load
    def load_config(self, config):
        # input
        self.vocab_size = config.vocab_size
        self.embed_size = config.embed_size
        self.length_max = config.length_max

        # hidden layer
        self.hidden_size = config.hidden_size
        self.hidden_num = config.hidden_num
        
        # dropout
        self.drop_out = config.drop_out


    # initial placeholder
    def initial_placeholder(self):
        self.input_sample = tf.placeholder(tf.int32, 
                                           [None, self.length_max])
        self.senti
    
    # model variable
    def model_variable(self):
        # embedding layer
        with tf.name_scope('embedding_layer'):
            embed_vocab = tf.get_variable('embedding', 
                                        [self.vocab_size, self.embed_size], 
                                        dtype=tf.float32)
            input_embed = tf.nn.embedding_lookup(embed, self.input)

        # softmax
        with tf.name_scope('mean_pooling'):
            out_put
        

    # build graph
    def build_graph(self):
        # lstm
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, 
                                                 forget_bias=0.0)
        lstm_cell_dropout = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, 
                                                          output_keep_prob=self.drop_out)
        self.multi_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_dropout] * self.hidden_num)
        

    
    # train model
    def model_train(self):
        pass
    
