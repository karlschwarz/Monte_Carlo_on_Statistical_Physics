import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from tqdm.auto import trange

import time
import pickle

import argparse

def set_logger(log_path):
    """
    From CS230-Stanford 
    https://github.com/cs230-stanford/cs230-code-examples/blob/master/tensorflow/vision/model/utils.py
    Sets the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

#beta_init = np.log(1 + np.sqrt(2)) / 2
beta_init = 1.0 / 2.5
linear_size = 32

parser = argparse.ArgumentParser(description='ising2d_metropolis')
parser.add_argument('--beta', '-b', default=beta_init, type=float, help='beta')
parser.add_argument('--j', '-j', default=1, type=int, help='j')
parser.add_argument('--size', '-s', default=linear_size, type=int, help='size')
parser.add_argument('--magnet', '-m', default=0, type=float, help='magnet')
args = parser.parse_args()

### fix the random seed
seed = 10
np.random.seed(seed)

global grids, dims, beta, j, h, delta

def parameters_init(dims_init, beta_init, j_init, h_init):
    global grids, dims, beta, j, h
    dims = (nb.int64(dims_init[0]), nb.int64(dims_init[1]))
    beta, j, h = nb.float64(beta_init), nb.int8(j_init), nb.float64(h_init)
    grids = np.reshape(np.random.choice([nb.int8(-1), nb.int8(1)], 
                                         size=dims_init[0]*dims_init[1]), dims_init)
    
######################################################################
### initialize the parameters
dims_init = (args.size, args.size)
beta_init, j_init, h_init = args.beta, args.j, args.magnet 
parameters_init(dims_init, beta_init, j_init, h_init)
######################################################################

######################################################################
types_neibor_get = nb.types.UniTuple(nb.types.UniTuple(nb.int8, 2), 4)(nb.types.UniTuple(nb.int8, 2))

@nb.cfunc(types_neibor_get)
def neibor_get_square(id_grid):
    height, width = dims
    height_index, width_index = id_grid
    l_neibor = (height_index, (width_index - 1) % width)
    r_neibor = (height_index, (width_index + 1) % width)
    u_neibor = ((height_index - 1) % height, width_index)
    d_neibor = ((height_index + 1) % height, width_index)          
    return (l_neibor, r_neibor, u_neibor, d_neibor)

types_energy_compute_one_grid = nb.float64(nb.int8[:, :], nb.types.UniTuple(nb.int8, 2), nb.int8)

@nb.njit(types_energy_compute_one_grid)
def energy_compute_one_grid(grids, id_grid, id_spin):
    energy_one_site = 0
    id_neibors = neibor_get_square(id_grid)
    for neibor in id_neibors:
        energy_one_site += -j * grids[neibor[0]][neibor[1]] * id_spin
    energy_one_site += -h * id_spin
    return energy_one_site

@nb.njit(nb.float64(nb.int8[:, :]))
def energy_compute_grids(grids):
    energy_total = 0
    for ii in range(dims[0]):
        for jj in range(dims[1]):
            id_neibors = neibor_get_square((nb.int8(ii), nb.int8(jj)))
            for ij in id_neibors:
                energy_ij = -1/2 * j * grids[ij[0]][ij[1]] * grids[ii][jj]
                energy_total += energy_ij
            ### compute the energy of external field
            energy_total += -h * grids[ii][jj]
    energy_per_spin = energy_total / (dims[0] * dims[1])
    return energy_per_spin

#@nb.njit(nb.float32(nb.int8[:, :]))
#def magnet_grids(grids):
#    m = 0
#    for ii in range(dims[0]*dims[1]):
#        row_grids, col_grids = ii // dims[1], ii % dims[1] 
#        m += grids[row_grids][col_grids]
#    m_avg = m / (dims[0] * dims[1])
#    return m_avg

#@nb.njit(nb.float32(nb.int8[:, :]))
#def magnet_grids(grids):
#    m = 0
#    for ii in range(dims[0]*dims[1]):
#        row_grids, col_grids = ii // dims[1], ii % dims[1] 
#        m += grids[row_grids][col_grids]
#    m_avg = m / (dims[0] * dims[1])
#    return m_avg

@nb.njit
def magnet_grids(grids):
    return np.mean(grids)

@nb.njit(nb.float64(nb.int8[:, :]))
def magnet_subgrids(grids):
    m_1, m_2 = 0, 0
    for i in range(dims[0]):
        for j in range(dims[1]):
            if (j + i) % 2 == 0:
                m_1 += grids[i][j]
            else:
                m_2 += grids[i][j]
    m_abs_per_spin = (np.abs(m_1) + np.abs(m_2)) / (dims[0] * dims[1])
    return m_abs_per_spin

@nb.njit(nb.int8[:, :](nb.int8[:, :]))
def one_site_mcmc(grids):
    id_random = np.random.randint(dims[0]*dims[1])
    id_height, id_width = nb.int8(id_random // dims[1]), nb.int8(id_random % dims[1])
    spin_old = grids[id_height][id_width]
    spin_new = nb.int8(-1 * spin_old)
    energy_old = energy_compute_one_grid(grids, (id_height, id_width), spin_old)
    energy_new = energy_compute_one_grid(grids, (id_height, id_width), spin_new)
    energy_delta = energy_new - energy_old
    if energy_delta <= 0:
        grids[id_height][id_width] = spin_new
    else:
        prob_accept = np.exp(-beta * energy_delta)
        if np.random.random() < prob_accept:
            grids[id_height][id_width] = spin_new
    return grids

@nb.njit(nb.int8[:, :](nb.int8[:, :]))
def one_step_mcmc(grids):
    for _ in range(dims[0]*dims[1]):
        grids = one_site_mcmc(grids)
    return grids

@nb.njit(nb.types.UniTuple(nb.float64[:], 2)(nb.int8[:, :], nb.int64))
def mcmc(grids, steps):
    energy_history = np.zeros(steps, dtype=np.float64)
    m_history = np.zeros(steps, dtype=np.float64)
    for i in range(steps):
        grids = one_step_mcmc(grids)
        energy_per_spin = energy_compute_grids(grids)
        m_per_spin = magnet_grids(grids)
        energy_history[i] = energy_per_spin
        m_history[i] = m_per_spin
    return energy_history, m_history
################################################################################

if __name__ == "__main__":
    
    ### Burning-in stage
    time_start = time.time()
    #energy_history, m_history = mcmc(grids, 50000)
    energy_history, m_history = mcmc(grids, 0)
    time_end = time.time()
    time_burn = time_end - time_start
    print(f"Time used in burning-in stage: {time_burn:.3f}.")
    ### Sampling stage
    time_start = time.time()
    _, m_history = mcmc(grids, 140000)
    time_end = time.time()
    print(f"Time spent on sampling of 2E5 samples: {time_end-time_start:.3f}.")
    print(f"<|M|> is: {np.mean(np.abs(m_history))}.")
    with open(f'../data_2d_ising/ising2d_square_size_{args.size}_beta_{beta:.3f}_h_{h:.3f}_pos_j_14E4_metropolis_v2.pkl', 'wb') as file:
        pickle.dump(m_history, file)
    
    