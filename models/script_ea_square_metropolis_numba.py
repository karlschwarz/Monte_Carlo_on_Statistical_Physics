import argparse
import os
from subprocess import check_call
import sys
from tqdm.auto import trange
import time

PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data_2d_ea/ea_square_size_16',
                    help="Directory for the data set.")
parser.add_argument('--grid_size', '-g_size', default=16, type=int, help='size of grid')
parser.add_argument('--ensemble_size', '-e_size', default=20, type=int, help='Size of J ensemble')

def launch_training_job(size, beta=10):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name
    Args:
        parent_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Launch training with this config
    cmd = "{python} ./ea_square_metropolis_numba.py -s {size} -b {beta}".format(python=PYTHON,\
                                                                                size=size,\
                                                                                beta=beta)
    print(cmd)
    check_call(cmd, shell=True)
    
if __name__ == "__main__":
    args = parser.parse_args()
    time_start = time.time()
    for _ in trange(args.ensemble_size):
        launch_training_job(args.grid_size)
    time_end = time.time()
    print(f"time used is: {time_end-time_start}.")