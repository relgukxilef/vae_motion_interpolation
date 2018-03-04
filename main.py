import argparse
import tensorflow as tf

from model import VAEMotionInterpolation

parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase', type=str, default='train', help='train or test')

args = parser.parse_args()

def main(_):
    tf.reset_default_graph()

    with tf.Session() as session:
        model = VAEMotionInterpolation(session)
        model.train() if args.phase == 'train' else model.test()#model.interpolate_video(1709, 1937, 60/24)

if __name__ == '__main__':
    tf.app.run()
