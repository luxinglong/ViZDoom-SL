import argparse
import tensorflow as tf

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

def bool_flag(string):
    """
    Parse boolean arguments from the command line.
    """
    if string.lower() in FALSY_STRINGS:
        return False
    elif string.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag. "
                                         "use 0 or 1")

def register_model_args(parser):
    """
    Register scenario parameters.
    """
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size for sl training")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="learning rate for sl training")
    parser.add_argument("--feature_img_height", type=int, default=60,
                        help="image height")
    parser.add_argument("--feature_img_width", type=int, default=108,
                        help="image width")
    parser.add_argument("--feature_img_channel", type=int, default=3,
                        help="image channel")
    parser.add_argument("--feature_vec_length", type=int, default=2,
                        help="the number of game variables")
                                                    
    parser.add_argument("--use_game_features", type=bool_flag, default=False,
                        help="whether use game features as auxiliary task")
    parser.add_argument("--use_game_variables", type=bool_flag, default=False,
                        help="whether use game variables as multimodal input")
    parser.add_argument("--use_lstm", type=bool_flag, default=False,
                        help="whether use lstm")
    parser.add_argument("--seq_len", type=int, default=1,
                        help="the seq length for lstm input")
    parser.add_argument("--game_features", type=str, default="enemy",
                        help="game features")
    parser.add_argument("--epoch", type=int, default=50,
                        help="number of epochs")
    parser.add_argument("--log_dir", type=str, default="./log",
                        help="the address of log dir")


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))

        tf.summary.histogram('histogram', var)
