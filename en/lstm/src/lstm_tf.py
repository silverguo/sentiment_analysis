import tensorflow as tf

__all__ = ['Senti_Lstm']

# lstm for sentiment analysis
class Senti_Lstm(object):

    # constructor
    def __init__(self, config, is_training=True):

        # load config
        # input
        vocab_size = config.vocab_size
        embed_size = config.embed_size
        step_num = config.step_num
        # hidden layer
        hidden_size = config.hidden_size
        hidden_num = config.hidden_num
        keep_prob = config.keep_prob
        # output
        num_class = config.num_class
        max_grad_norm = config.max_grad_norm
        # train param
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size


        # initial placeholder
        # input sentence
        self.input_sample = tf.placeholder(tf.int32, 
                                           [self.batch_size, None])
        # sentiment label
        self.real_label = tf.placeholder(tf.int32, [self.batch_size])


        # lstm layer
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, 
                                                 forget_bias=0.0, 
                                                 state_is_tuple=True)
        if is_training and keep_prob < 1:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, 
                                                      output_keep_prob=keep_prob)
        multi_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell] * hidden_num, 
                                                 state_is_tuple=True)
        self._initial_state = multi_lstm.zero_state(self.batch_size, tf.float32)
        
        # embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding_layer'):
            embed_vocab = tf.get_variable('embedding', 
                                          [vocab_size, embed_size], 
                                          dtype=tf.float32)
            input_embed = tf.nn.embedding_lookup(embed_vocab, 
                                                 self.input_sample)
            if is_training and keep_prob < 1:
                input_embed = tf.nn.dropout(input_embed, keep_prob)

        # build lstm
        with tf.variable_scope('lstm_layer'):
            output_lstm, final_state = tf.nn.dynamic_rnn(cell=multi_lstm, 
                                                        inputs=input_embed, 
                                                        sequence_length=self.seq_length, 
                                                        initial_state=self._initial_state)
            # output flat
            output_lstm_flat = tf.reshape(output_lstm, [-1, hidden_size])
        

        # softmax
        with tf.name_scope('softmax_layer'):
            w_softmax = tf.get_variable('w_softmax', 
                                        [hidden_size, num_class], 
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_softmax = tf.get_variable('b_softmax', 
                                        [num_class], 
                                        initializer=tf.constant_initializer(0.0))
            self.logits = tf.matmul(output_lstm_flat, w_softmax) + b_softmax
            self.prob_flat = tf.nn.softmax(self.logits)
            self.pred_label = tf.argmax(self.logits, 1)
        

        # loss
        with th.name_scope('loss'):
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, 
                                                                       self.real_label)
            self.cost = tf.reduce_mean(self.loss)

        
        # accuracy
        with tf.name_scope('accuracy'):
            self.correct_pred = tf.equal(self.pred_label, real_label)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32), 
                                           name='accuracy')
        
        
        # add summary
        loss_summary = tf.summary.scalar('loss', self.cost)
        accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)

        # parameter training
        if is_training:
            params = tf.trainable_variables()
            with tf.name_scope('train'):
                opt = tf.train.AdamOptimizer(learning_rate)
            gradients = tf.gradients(self.cost, params)
            clipped_grad, norm = tf.clip_by_global_norm(gradients, max_grad_norm)
            self.train_opt = opt.apply_gradients(zip(clipped_grad, params))
            # norm
            grad_summary = tf.summary.scalar('grad_norm', norm)
        

        # summary merge
        self.merged_summary = tf.summary.merge_all()
        

        # param update
        self.new_batch_size = tf.placeholder(tf.int32, shape=[], 
                                             name='new_batch_size')
        self._batch_size_update = tf.assign(self.batch_size, self.new_batch_size)
        self.new_learning_rate = tf.placeholder(tf.float32, shape=[], 
                                                name='new_learning_rate')
        self._learning_rate_update = tf.assign(self.learning_rate, 
                                               self.new_learning_rate)

        return


    # update learning rate
    def assign_new_learning_rate(self, session, lr):
        session.run(self._learning_rate_update, 
                    feed_dict={self.new_learning_rate: lr})


    # update batch size
    def assign_new_batch_size(self, session, bs):
        session.run(self._batch_size_update, 
                    feed_dict={self.new_batch_size: bs})


        
    
