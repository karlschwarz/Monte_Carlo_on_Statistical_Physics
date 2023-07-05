using Distributions
using LinearAlgebra
using Einsum

function parameters_init(dims_init, beta_init, h_init)
    global subgrid_1, subgrid_2, grid_3d, dims, beta, h
    dims = (dims_init[1], dims_init[2], dims_init[3])
    beta, h = beta_init, h_init
    p_binomial = Binomial(1, 1/2)
    subgrid_1 = rand(p_binomial, dims_init[2]*dims_init[3])
    subgrid_1 = reshape(subgrid_1, (dims_init[2], dims_init[3]))
    subgrid_1 = (-1) .^ subgrid_1
    #######################################
    #subgrid_1 = ones((dims_init[2], dims_init[3]))
    #subgrid_1 = [
    #    [-1 1 1];
    #    [1 -1 -1];
    #    [-1 1 -1];
    #]
    #######################################
    subgrid_2 = rand(p_binomial, dims_init[2]*dims_init[3])
    subgrid_2 = reshape(subgrid_2, (dims_init[2], dims_init[3]))
    subgrid_2 = (-1) .^ subgrid_2
    #######################################
    #subgrid_2 = -1 .* ones((dims_init[2], dims_init[3]))
    #subgrid_2 = [
    #    [-1 -1 1];
    #    [1 1 -1];
    #    [-1 -1 -1];
    #]
    #######################################
    #######################################
    #subgrid_3 = [
    #    [-1 1 1];
    #    [-1 -1 1];
    #    [1 1 -1];
    #]
    grid_3d = rand(p_binomial, dims_init[1]*dims_init[2]*dims_init[3])
    grid_3d = reshape(grid_3d, (dims_init[1], dims_init[2], dims_init[3]))
    grid_3d = (-1) .^ grid_3d
    #######################################
    #grid_3d[1, :, :] = ones((dims_init[2], dims_init[3]))
    #grid_3d[2, :, :] = -1 .* ones((dims_init[2], dims_init[3]))
    #grid_3d[3, :, :] = ones((dims_init[2], dims_init[3]))
    #grid_3d = cat(copy(subgrid_1), copy(subgrid_2), copy(subgrid_3), dims=3)
    #grid_3d = permutedims(grid_3d, (3, 1, 2))
    #######################################
end

function neibor_get(j_matrix, id_grid)
    height_index, width_index = id_grid
    j_vector = j_matrix[height_index, width_index, :]
    l_neibor = (height_index, width_index - 1)
    d_neibor = (height_index + 1, width_index)                    
    r_neibor = (height_index, width_index + 1)
    u_neibor = (height_index - 1, width_index)
    jl, jd, jr, ju = j_vector[1], j_vector[2], j_vector[3], j_vector[4] 
    if height_index == 1
        if width_index == 1
            neibors_tuple = ((d_neibor, r_neibor, (1, 1), (1, 1)), 2)
            j_tuple = (jd, jr, 0, 0)
        elseif width_index == dims[3] 
            neibors_tuple = ((l_neibor, d_neibor, (1, 1), (1, 1)), 2)
            j_tuple = (jl, jd, 0, 0)
        else
            neibors_tuple = ((l_neibor, d_neibor, r_neibor, (1, 1)), 3)
            j_tuple = (jl, jd, jr, 0)
        end
    elseif height_index == dims[2]
        if width_index == 1
            neibors_tuple = ((r_neibor, u_neibor, (1, 1), (1, 1)), 2)
            j_tuple = (jr, ju, 0, 0)
        elseif width_index == dims[3]
            neibors_tuple = ((l_neibor, u_neibor, (1, 1), (1, 1)), 2)
            j_tuple = (jl, ju, 0, 0)
        else
            neibors_tuple = ((l_neibor, r_neibor, u_neibor, (1, 1)), 3)
            j_tuple = (jl, jr, ju, 0)
        end
    else
        if width_index == 1
            neibors_tuple = ((d_neibor, r_neibor, u_neibor, (1, 1)), 3)
            j_tuple = (jd, jr, ju, 0)
        elseif width_index == dims[3]
            neibors_tuple = ((l_neibor, d_neibor, u_neibor, (1, 1)), 3)
            j_tuple = (jl, jd, ju, 0)
        else
            neibors_tuple = ((l_neibor, d_neibor, r_neibor, u_neibor), 4)
            j_tuple = (jl, jd, jr, ju)
        end
    end
    return neibors_tuple, j_tuple
end

function energy_compute_site(subgrid, j_matrix, id_grid, id_spin)
    energy_one_site = 0
    (id_neibors, num_neibors), j_tuple = neibor_get(j_matrix, id_grid)
    id_row, id_col = id_grid     
    for i in 1: num_neibors
        neibor = id_neibors[i]
        energy_one_site += -j_tuple[i] * subgrid[neibor[1], neibor[2]] * id_spin
    end
    energy_one_site += -h * id_spin
    return energy_one_site
end

function energy_compute_grid(subgrid, j_matrix)
    energy_total = 0
    for ii in 1: dims[2]
        for jj in 1: dims[3]
            (id_neibors, num_neibors), j_tuple = neibor_get(j_matrix, (Int8(ii), Int8(jj)))
            for k in 1: num_neibors
                ij = id_neibors[k]
                energy_ij = -1/2 * j_tuple[k] * subgrid[ij[1], ij[2]] * subgrid[ii, jj]
                energy_total += energy_ij
            end
            ### compute the energy of external field
            energy_total += -h * subgrid[ii, jj]
        end
    end
    #print(energy_total)
    energy_per_spin = energy_total / (dims[2] * dims[3])
    return energy_per_spin
end

function energy_compute_grid_3d(subgrid, j_matrix, id_layer)
    energy_total = 0
    for ii in 1: dims[2]
        for jj in 1: dims[3]
            (id_neibors, num_neibors), j_tuple = neibor_get(j_matrix, (ii, jj))
            for k in 1: num_neibors
                ij = id_neibors[k]
                energy_ij = -1/2 * j_tuple[k] * subgrid[ij[1], ij[2]] * subgrid[ii, jj]
                energy_total += energy_ij
            end
            if id_layer == 1 
                energy_total += -j_matrix[ii, jj, 5] * grid_3d[id_layer + 1, ii, jj] * subgrid[ii, jj]
                #print(id_layer)
            elseif id_layer == dims[1]
                energy_total += -j_matrix[ii, jj, 5] * grid_3d[id_layer - 1, ii, jj] * subgrid[ii, jj]
                #print(id_layer)
            else
                energy_total += -j_matrix[ii, jj, 5] * grid_3d[id_layer - 1, ii, jj] * subgrid[ii, jj]
                energy_total += -j_matrix[ii, jj, 6] * grid_3d[id_layer + 1, ii, jj] * subgrid[ii, jj]
                #print(id_layer)
            end
            ### compute the energy of external field
            energy_total += -h * subgrid[ii, jj]
        end
    end
    energy_per_spin = energy_total / (dims[2] * dims[3])
    return energy_per_spin
end

function energy_one_layer(subgrid, j_matrix, id_layer)
    energy_total = 0
    for ii in 1: dims[2]
        for jj in 1: dims[3]
            (id_neibors, num_neibors), j_tuple = neibor_get(j_matrix, (Int8(ii), Int8(jj)))
            for k in 1: num_neibors
                ij = id_neibors[k]
                energy_ij = -1 / 2 * j_tuple[k] * subgrid[ij[1], ij[2]] * subgrid[ii, jj]
                energy_total += energy_ij
            end
            if id_layer == 1
                energy_total += -j_matrix[ii, jj, 5] * grid_3d[id_layer + 1, ii, jj] * subgrid[ii, jj] / 2
                #print(id_layer)
            elseif id_layer == dims[1]
                energy_total += -j_matrix[ii, jj, 5] * grid_3d[id_layer - 1, ii, jj] * subgrid[ii, jj] / 2
                #print(id_layer)
            else
                energy_total += -j_matrix[ii, jj, 5] * grid_3d[id_layer - 1, ii, jj] * subgrid[ii, jj] / 2
                energy_total += -j_matrix[ii, jj, 6] * grid_3d[id_layer + 1, ii, jj] * subgrid[ii, jj] / 2
                #print(id_layer)
            ### compute the energy of external field
            end
            energy_total += -h * subgrid[ii, jj]
        end
    end
    return energy_total
end

function energy_total(grid_3d)
    num_layer = dims[1]
    energy_sum = 0
    for i in 1: num_layer
        j_matrix = j_matrix_array[i, :, :, :]
        subgrid = grid_3d[i, :, :]
        energy_sum += energy_one_layer(subgrid, j_matrix, i)
    end
    return energy_sum
end

function magnet_grid(grid)
    m_mean = mean(grid)
    return m_mean
end


