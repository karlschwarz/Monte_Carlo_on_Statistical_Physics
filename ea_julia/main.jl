include("./coupling_create.jl")
include("./utils.jl")
include("./tensor_prepare.jl")
include("tnmc_process.jl")

########################################
dims_init = (3, 4, 4)
num_j = 6
beta_init = 0.2
h_init = 0
bond_dims = 2

j_matrix_array = j_matrix_3d(dims_init, num_j)
parameters_init(dims_init, beta_init, h_init)
tensors_init(j_matrix_array)

grid_3d, prob_acc_list = full_evv_3d_onestep(grid_3d, j_matrix_array, bond_dims)