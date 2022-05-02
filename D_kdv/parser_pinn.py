import argparse
from datetime import datetime
import math

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--maxiter', default=50000, type=int
    )
    parser.add_argument(
        '--resample_interval', default=1000, type=int
    )
    parser.add_argument(
        '--sample_num', default=1000, type=int
    )
    parser.add_argument(
        '--freq_draw', default=5, type=int
    )
    parser.add_argument(
        '--resample_N', default=100
    )
    parser.add_argument(
        '--net_seq', default=[2, 64, 64, 64, 64, 1]
    )
    parser.add_argument(
        '--save_path', default=f'./data/{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
    )


    parser.add_argument(
        '--verbose', default=False
    )
    parser.add_argument(
        '--sigma', default=0.1
    )
    parser.add_argument(
        '--repeat', default=50
    )
    parser.add_argument(
        '--start_epoch', default=0
    )
    parser.add_argument(
        '--c', default=7.0
    )
    parser.add_argument(
        '--Tmax', default=2.0
    )
    parser.add_argument(
        '--half_L', default=4*math.pi
    )
    return parser
