import os
import sys
import argparse

from src.agent import Agent
from src.utils import register_model_args

import tensorflow as tf

def main():
    parser = argparse.ArgumentParser(description='LUBAN runner')
    register_model_args(parser) 
    params, unparsed = parser.parse_known_args(sys.argv)
    sess = tf.Session()
    agent = Agent(sess ,params)
    agent.train(checkpoint_dir="./checkpoint", data_dir='./data/dataset-50-3-2.hdf5')


if __name__ == '__main__':
    main()
