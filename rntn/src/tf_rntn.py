import tensorflow as tf

__all__ = ['model_train']

# flags of tensorflow
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('u_weight', 0.0001, 
                   'Range for random weight')
flags.DEFINE_integer('word_size', 10, 
                     'Word vector size')
flags.DEFINE_integer('label_number', 5, 
                     'number of labels')

def weight_variable(shape, name=None):
    return tf.Variable(tf.random_normal(shape, 0, 
                                        FLAGS.u_weight), 
                       name=name)

def bias_variables(shape, name=None):
    return tf.Variable(tf.zeros(shape, dtype=tf.float32), 
                       name=name)

def model_train():
    rntn = tf.VariableScope(False, name='rntn')

    # define parameters of rntn
    with tf.name_scope('weights'):
        tensorV = weight_variable([2 * FLAGS.word_size, 
                                   2 * FLAGS.word_size, 
                                   FLAGS.word_size], 
                                  name='tensorV')
        linearW = weight_variable([FLAGS.word_size, 
                                   2 * FLAGS.word_size], 
                                  name='linearW')
        softW = weight_variable([FLAGS.label_number, 
                                 FLAGS.word_size], 
                                name='softW')
    with tf.name_scope('bias'):
        linearB = bias_variables([FLAGS.word_size, 1], 
                                 name='linearB')
        softB = bias_variables([FLAGS.label_number, 1], 
                               name='softB')

    # load


    return