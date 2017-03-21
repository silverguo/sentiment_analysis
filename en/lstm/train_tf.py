from src import *
import tensorflow as tf
import pickle

flags =tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('vocab_size', 200000, 'vocabulary size')
flags.DEFINE_integer('embed_size', 300, 'embedding dimension')
flags.DEFINE_integer('step_num', 50, 'rnn step number')
flags.DEFINE_integer('hidden_size', 128, 'lstm hidden neural size')
flags.DEFINE_integer('hidden_num', 1, 'hidden layer number')
flags.DEFINE_float('keep_prob', 0.5, 'dropout rate')
flags.DEFINE_integer('num_class', 2, 'class number')
flags.DEFINE_integer('max_grad_norm', 5, 'max gradient norm')
flags.DEFINE_float('learning_rate', 0.1, 'learning rate')
flags.DEFINE_integer('batch_size', 64, 'training batch size')
flags.DEFINE_string('dataset_path', './demo/data/imdb_input.pickle', 'dataset path')
flags.DEFINE_string('model_dir', './demo/model/', 'model directory')
flags.DEFINE_integer('check_point_every', 2, 'number of epoch between checkpoint')
flags.DEFINE_float('init_scale', 0.1, 'initial scale')
flags.DEFINE_integer('epoch_num', 10, 'number of epoch')
flags.DEFINE_integer('epoch_decay', 6, 'epoch start lr decay')
flags.DEFINE_float('learning_rate_decay', 0.5, 'learning rate decay')


# param config
class Config(object):

    vocab_size = FLAGS.vocab_size
    embed_size = FLAGS.embed_size
    step_num = FLAGS.step_num
    hidden_size = FLAGS.hidden_size
    hidden_num = FLAGS.hidden_num
    keep_prob = FLAGS.keep_prob
    num_class = FLAGS.num_class
    max_grad_norm = FLAGS.max_grad_norm
    learning_rate = FLAGS.learning_rate
    batch_size = FLAGS.batch_size
    dataset_path = FLAGS.dataset_path
    model_dir = FLAGS.model_dir
    check_point_every = FLAGS.check_point_every
    init_scale = FLAGS.init_scale
    epoch_num = FLAGS.epoch_num
    epoch_decay = FLAGS.epoch_decay
    learning_rate_decay = FLAGS.learning_rate_decay


# run epoch 
def run_epoch(session, model, dataset):
    for input_sample, seq_length, real_label in dataset:

        # feed dict
        feed_dict = {}
        feed_dict[model.input_sample] = input_sample
        feed_dict[model.seq_length] = seq_length
        feed_dict[model.real_label] = real_label

        # param fetch
        fetches = {'cost': model.cost,
                   'accuracy': model.accuracy, 
                   'summary': model.summary}

        # initial state
        state = session.run(model._initial_state)
        for idx, (c, h) in enumerate(model._initial_state):
            feed_dict[c] = state[idx].c
            feed_dict[h] = state[idx].h
        
        # run model
        cost, accuracy, summary = session.run(fetches, feed_dict)

        # show accuracy
        print('accuracy: %.3f' % (accuracy))

    return cost



# main
def main(_):

    # load config of train and eval
    print('loading the dataset and config')
    train_config = Config()
    eval_config = Config()
    # no dropout for evaluation
    eval_config.keep_prob = 1.0

    # load data
    with open(train_config.dataset_path, 'rb') as f:
        sample_dict = pickle.load(f)
    
    # training
    print('training start')
    with tf.Graph().as_default():
        # default initializer
        initializer = tf.random_uniform_initializer(-train_config.init_scale, 
                                                    train_config.init_scale)
        
        # name scope for train and eval
        with tf.name_scope('train'):
            with tf.variable_scope('model', reuse=None, initializer=initializer):
                mtrain = Senti_Lstm(is_training=True, config=train_config)
            tf.summary.scalar('training loss', mtrain.cost)
            tf.summary.scalar('learning rate', mtrain.learning_rate)
        
        with tf.name_scope('valid'):
            with tf.variable_scope('model', reuse=True, initializer=initializer):
                mvalid = Senti_Lstm(is_training=False, config=eval_config)
            tf.summary.scalar('validation Loss', mvalid.cost)
        
        with tf.name_scope('test'):
            with tf.variable_scope('model', reuse=True, initializer=initializer):
                mtest = Senti_Lstm(is_training=False, config=eval_config)
            tf.summary.scalar('validation Loss', mtest.cost)
        
        # supervisor for checkpoint the model
        sv = tf.train.Supervisor(logdir=config.model_dir)
        with sv.managed_session() as session:
            # train epoch
            for i in range(config.epoch_num):
                print('the %d training epoch'%i)
                # learning rate decay
                learning_rate_decay = config.learning_rate_decay ** \
                                      max(i - config.epoch_decay, 0.0)
                m.assign_new_learning_rate(session, 
                                           config.learning_rate * learning_rate_decay)
                # run epoch
                train_cost = run_epoch(session, m, sample_dict['train'])
                print('epoch %d train cost: %.3f' % (i, train_cost))
                valid_cost = run_epoch(session, m, sample_dict['valid'])
                print('epoch %d valid cost: %.3f' % (i, valid_cost))
                
                # checkpoint
                if i % config.check_point_every == 0:
                    print('model chechpoint to {}'.format(config.model_dir))
                    sv.saver.save(session, config.model_dir, 
                                  global_step=sv.global_step)
            
            # test in end of train
            test_cost = run_epoch(session, m, sample_dict['test'])
            print('train finish test cost: %.3f' % (test_cost))



if __name__ == '__main__':
    tf.app.run()
    
