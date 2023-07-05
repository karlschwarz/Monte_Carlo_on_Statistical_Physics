using Distributions
using LinearAlgebra
using Einsum

function B_tensor_prepare(j_vec)
    j_vec_len = length(j_vec)
    B_list = Any[]
    for i in 1: j_vec_len
        jj = j_vec[i]
        if jj == 1
            push!(B_list, B_pos)
        else
            push!(B_list, B_neg)
        end
    end
    return B_list
end

function bdy_vector_prepare(subgrid, j_matrix, id_loc, pos_index)
    id_row, id_col = id_loc
    if subgrid[id_row, id_col] == 1
        if j_matrix[id_row, id_col, pos_index] == 1
            bdy_vector = spin_pos_up
        else
            bdy_vector = spin_neg_up
        end
    else
        if j_matrix[id_row, id_col, pos_index] == 1
            bdy_vector = spin_pos_down
        else
            bdy_vector = spin_neg_down
        end
    end
    return bdy_vector
end

#function parameters_init(dims_init, beta_init, h_init)
#    global subgrid_1, subgrid_2, grid_3d, dims, beta, h
#    dims = (dims_init[1], dims_init[2], dims_init[3])
#    beta, h = beta_init, h_init
#    p_binomial = Binomial(1, 1/2)
#    subgrid_1 = rand(p_binomial, dims_init[2]*dims_init[3])
#    subgrid_1 = reshape(subgrid_1, (dims_init[2], dims_init[3]))
#    subgrid_1 = (-1) .^ subgrid_1
#    subgrid_2 = rand(p_binomial, dims_init[2]*dims_init[3])
#    subgrid_2 = reshape(subgrid_2, (dims_init[2], dims_init[3]))
#    subgrid_2 = (-1) .^ subgrid_2
#    grid_3d = rand(p_binomial, dims_init[1]*dims_init[2]*dims_init[3])
#    grid_3d = reshape(grid_3d, (dims_init[1], dims_init[2], dims_init[3]))
#    grid_3d = (-1) .^ grid_3d
#end


function node_tensor_create()
    B_pos = sqrt(Complex[exp(beta) exp(-beta); exp(-beta) exp(beta)])
    B_neg = sqrt(Complex[exp(-beta) exp(beta); exp(beta) exp(-beta)])
    spin_up = Complex[1, 0]
    spin_down = Complex[0, 1]
    @einsum spin_pos_up[j] := spin_up[i] * B_pos[i, j]
    @einsum spin_pos_down[j] := spin_down[i] * B_pos[i, j]
    @einsum spin_neg_up[j] := spin_up[i] * B_neg[i, j]
    @einsum spin_neg_down[j] := spin_down[i] * B_neg[i, j]
    return B_pos, B_neg, spin_pos_up, spin_pos_down, spin_neg_up, spin_neg_down
end

function copy_tensor_create()
    I_2_array = zeros(Complex, (2, 2))
    I_3_array = zeros(Complex, (2, 2, 2))
    I_4_array = zeros(Complex, (2, 2, 2, 2))
    I_5_array = zeros(Complex, (2, 2, 2, 2, 2))
    I_6_array = zeros(Complex, (2, 2, 2, 2, 2, 2))
    I_2_up_array = zeros(Complex, (2, 2))
    I_2_down_array = zeros(Complex, (2, 2))
    I_3_up_array = zeros(Complex, (2, 2, 2))
    I_3_down_array = zeros(Complex, (2, 2, 2))
    I_up_array = zeros(Complex, (2, 2, 2, 2))
    I_down_array = zeros(Complex, (2, 2, 2, 2))
    I_4_up_array = zeros(Complex, (2, 2, 2, 2))
    I_4_down_array = zeros(Complex, (2, 2, 2, 2))    
    I_5_up_array = zeros(Complex, (2, 2, 2, 2, 2))
    I_5_down_array = zeros(Complex, (2, 2, 2, 2, 2))
    I_6_up_array = zeros(Complex, (2, 2, 2, 2, 2, 2))
    I_6_down_array = zeros(Complex, (2, 2, 2, 2, 2, 2))
    for i in 1: 2
        if i == 1
            I_2_array[i, i] = exp(beta * h)
            I_3_array[i, i, i] = exp(beta * h)
            I_4_array[i, i, i, i] = exp(beta * h)
            I_5_array[i, i, i, i, i] = exp(beta * h)
            I_6_array[i, i, i, i, i, i] = exp(beta * h)
        else
            I_2_array[i, i] = exp(-beta * h)
            I_3_array[i, i, i] = exp(-beta * h)
            I_4_array[i, i, i, i] = exp(-beta * h)
            I_5_array[i, i, i, i, i] = exp(-beta * h)
            I_6_array[i, i, i, i, i, i] = exp(-beta * h)
        end
    end
    I_up_array[1, 1, 1, 1] = exp(beta * h)
    I_down_array[2, 2, 2, 2] = exp(-beta * h)
    I_3_up_array[1, 1, 1] = exp(beta * h)
    I_3_down_array[2, 2, 2] = exp(-beta * h)
    I_2_up_array[1, 1] = exp(beta * h)
    I_2_down_array[2, 2] = exp(-beta * h)
    I_4_up_array[1, 1, 1, 1] = exp(beta * h)
    I_4_down_array[2, 2, 2, 2] = exp(-beta * h)    
    I_5_up_array[1, 1, 1, 1, 1] = exp(beta * h)
    I_5_down_array[2, 2, 2, 2, 2] = exp(-beta * h)    
    I_6_up_array[1, 1, 1, 1, 1, 1] = exp(beta * h)
    I_6_down_array[2, 2, 2, 2, 2, 2] = exp(-beta * h)
    I_1 = Complex[exp(beta * h), exp(-beta * h)]
    #I_2 = Complex(I_2_array, dtype=torch.complex128, device=my_device)
    #I_3 = torch.tensor(I_3_array, dtype=torch.complex128, device=my_device)
    #I_4 = torch.tensor(I_4_array, dtype=torch.complex128, device=my_device)
    #I_5 = torch.tensor(I_5_array, dtype=torch.complex128, device=my_device)
    #I_6 = torch.tensor(I_6_array, dtype=torch.complex128, device=my_device)
    #I_up = torch.tensor(I_up_array, dtype=torch.complex128, device=my_device)
    #I_down = torch.tensor(I_down_array, dtype=torch.complex128, device=my_device)
    #I_3_up = torch.tensor(I_3_up_array, dtype=torch.complex128, device=my_device)
    #I_3_down = torch.tensor(I_3_down_array, dtype=torch.complex128, device=my_device)
    #I_2_up = torch.tensor(I_2_up_array, dtype=torch.complex128, device=my_device)
    #I_2_down = torch.tensor(I_2_down_array, dtype=torch.complex128, device=my_device)
    #I_4_up = torch.tensor(I_4_up_array, dtype=torch.complex128, device=my_device)
    #I_4_down = torch.tensor(I_4_down_array, dtype=torch.complex128, device=my_device)
    #I_5_up = torch.tensor(I_5_up_array, dtype=torch.complex128, device=my_device)
    #I_5_down = torch.tensor(I_5_down_array, dtype=torch.complex128, device=my_device)
    #I_6_up = torch.tensor(I_6_up_array, dtype=torch.complex128, device=my_device)
    #I_6_down = torch.tensor(I_6_down_array, dtype=torch.complex128, device=my_device)
    copy_array = (
        I_2_array, I_3_array, I_4_array, 
        I_5_array, I_6_array, I_up_array, 
        I_down_array, I_3_up_array, I_3_down_array, 
        I_2_up_array, I_2_down_array, I_4_up_array, 
        I_4_down_array, I_5_up_array, I_5_down_array, 
        I_6_up_array, I_6_down_array
    )
    return copy_array
end

function row_tensor_mpo_bulk_layer_create(j_matrix, row_id)
    row_tensor_mpo = Any[]
    j_vec_left = [
        j_matrix[row_id, 1, 2], j_matrix[row_id, 1, 3], 
        j_matrix[row_id, 1, 4], j_matrix[row_id, 1, 5], 
        j_matrix[row_id, 1, 6]
    ]
    j_vec_right = [
        j_matrix[row_id, dims[3], 1], j_matrix[row_id, dims[3], 2], 
        j_matrix[row_id, dims[3], 4], j_matrix[row_id, dims[3], 5], 
        j_matrix[row_id, dims[3], 6]
    ]
    size_j_vec_left = size(j_vec_left)[1]
    size_j_vec_right = size(j_vec_right)[1]
    B_array_left = zeros(Complex, (size_j_vec_left, 2, 2))
    B_array_right = zeros(Complex, (size_j_vec_right, 2, 2))
    for j in 1: size_j_vec_left
        jj = j_vec_left[j]
        if jj == 1
            B_array_left[j, :, :] = B_pos
        else
            B_array_left[j, :, :] = B_neg
        end
    end
    for j in 1: size_j_vec_right
        jj = j_vec_right[j]
        if jj == 1
            B_array_right[j, :, :] = B_pos
        else
            B_array_right[j, :, :] = B_neg
        end
    end
    B_1_left, B_2_left = B_array_left[1, :, :], B_array_left[2, :, :]
    B_3_left, B_4_left = B_array_left[3, :, :], B_array_left[4, :, :]
    B_1_right, B_2_right = B_array_right[1, :, :], B_array_right[2, :, :]
    B_3_right, B_4_right = B_array_right[3, :, :], B_array_right[4, :, :]  
    B_5_left, B_5_right = B_array_left[5, :, :], B_array_right[5, :, :]
    @einsum A_5_left[i, k, m, o, q] := B_1_left[i, j] * B_2_left[k, l] * B_3_left[m, n] * B_4_left[o, p] * B_5_left[q, v] * I_5[j, l, n, p, v]
    @einsum A_5_right[i, k, m, o, q] := B_1_right[i, j] * B_2_right[k, l] * B_3_right[m, n] * B_4_right[o, p] * B_5_right[q, v] * I_5[j, l, n, p, v]
    A_5_left = reshape(A_5_left, (1, 2, 2, 2, 2, 2))
    A_5_right = reshape(A_5_right, (2, 2, 1, 2, 2, 2))
    push!(row_tensor_mpo, A_5_left)
    for col_id in 2: dims[2]-1
        j_vec = j_matrix[row_id, col_id, :]
        len_j_vec = size(j_vec)[1]
        B_array = zeros(Complex, (len_j_vec, 2, 2))
        for j in 1: len_j_vec
            jj = j_vec[j]
            if jj == 1
                B_array[j, :, :] = B_pos
            else
                B_array[j, :, :] = B_neg
            end
        end
        B_1, B_2 = B_array[1, :, :], B_array[2, :, :]
        B_3, B_4 = B_array[3, :, :], B_array[4, :, :]
        B_5, B_6 = B_array[5, :, :], B_array[6, :, :]
        @einsum A_6_i[i, k, m, o, q, r] := B_1[i, j] * B_2[k, l] * B_3[m, n] * B_4[o, p] * B_5[q, v] * B_6[r, s] * I_6[j, l, n, p, v, s]
        #A_6_i = A_6_i / A_6_i.norm()
        push!(row_tensor_mpo, A_6_i)
    end
    push!(row_tensor_mpo, A_5_right)
    return row_tensor_mpo
end

function row_tensor_mpo_bdy_layer_create(j_matrix, row_id)
    row_tensor_mpo = Any[]
    j_vec_left = [
        j_matrix[row_id, 1, 2], j_matrix[row_id, 1, 3], 
        j_matrix[row_id, 1, 4], j_matrix[row_id, 1, 5]
    ]
    j_vec_right = [
        j_matrix[row_id, dims[3], 1], j_matrix[row_id, dims[3], 2], 
        j_matrix[row_id, dims[3], 4], j_matrix[row_id, dims[3], 5]
    ]
    size_j_vec_left = size(j_vec_left)[1]
    size_j_vec_right = size(j_vec_right)[1]
    B_array_left = zeros(Complex, (size_j_vec_left, 2, 2))
    B_array_right = zeros(Complex, (size_j_vec_right, 2, 2))   
    for j in 1: size_j_vec_left 
        jj = j_vec_left[j]
        if jj == 1
            B_array_left[j, :, :] = B_pos
        else
            B_array_left[j, :, :] = B_neg
        end
    end
    for j in 1: size_j_vec_right
        jj = j_vec_right[j]
        if jj == 1
            B_array_right[j, :, :] = B_pos
        else
            B_array_right[j, :, :] = B_neg
        end
    end
    B_1_left, B_2_left = B_array_left[1, :, :], B_array_left[2, :, :]
    B_3_left, B_4_left = B_array_left[3, :, :], B_array_left[4, :, :]
    B_1_right, B_2_right = B_array_right[1, :, :], B_array_right[2, :, :]
    B_3_right, B_4_right = B_array_right[3, :, :], B_array_right[4, :, :]
    @einsum A_4_left[i, k, m, o] := B_1_left[i, j] * B_2_left[k, l] * B_3_left[m, n] * B_4_left[o, p] * I_4[j, l, n, p]
    @einsum A_4_right[i, k, m, o] := B_1_right[i, j] * B_2_right[k, l] * B_3_right[m, n] * B_4_right[o, p] * I_4[j, l, n, p]
    A_4_left = reshape(A_4_left, (1, 2, 2, 2, 2))
    A_4_right = reshape(A_4_right, (2, 2, 1, 2, 2))
    push!(row_tensor_mpo, A_4_left)
    for col_id in 2: dims[3]-1
        B_array = zeros(Complex, (6, 2, 2))
        j_vec = j_matrix[row_id, col_id, 1:6]
        for j in 1: 6
            jj = j_vec[j]
            if jj == 1
                B_array[j, :, :] = B_pos
            else
                B_array[j, :, :] = B_neg
            end
        end
        B_1, B_2 = B_array[1, :, :], B_array[2, :, :]
        B_3, B_4 = B_array[3, :, :], B_array[4, :, :]
        B_5 = B_array[5, :, :]
        @einsum A_5_i[i, k, m, o, q] := B_1[i, j] * B_2[k, l] * B_3[m, n] * B_4[o, p] * B_5[q, v] * I_5[j, l, n, p, v]
        #A_5_i = A_5_i / A_5_i.norm()
        push!(row_tensor_mpo, A_5_i)
    end
    push!(row_tensor_mpo, A_4_right)
    return row_tensor_mpo
end

function up_row_tensor_mps_bulk_layer_create(j_matrix)
    row_tensor_mps = Any[]
    j_vec_left = [
        j_matrix[1, 1, 2], j_matrix[1, 1, 3], 
        j_matrix[1, 1, 5], j_matrix[1, 1, 6]
    ]
    j_vec_right = [
        j_matrix[1, dims[3], 1], j_matrix[1, dims[3], 2], 
        j_matrix[1, dims[3], 5], j_matrix[1, dims[3], 6]
    ]
    B_array_left = zeros(Complex, (4, 2, 2))
    B_array_right = zeros(Complex, (4, 2, 2))
    for i in 1: 4
        jj_left = j_vec_left[i]
        jj_right = j_vec_right[i]
        if jj_left == 1
            B_array_left[i, :, :] = B_pos
        else
            B_array_left[i, :, :] = B_neg
        end
        if jj_right == 1
            B_array_right[i, :, :] = B_pos
        else
            B_array_right[i, :, :] = B_neg
        end
    end
    B_1_left, B_2_left = B_array_left[1, :, :], B_array_left[2, :, :]
    B_3_left, B_4_left = B_array_left[3, :, :], B_array_left[4, :, :]
    B_1_right, B_2_right = B_array_right[1, :, :], B_array_right[2, :, :]
    B_3_right, B_4_right = B_array_right[3, :, :], B_array_right[4, :, :]
    @einsum A_4_left[i, k, m, o] := B_1_left[i, j] * B_2_left[k, l] * B_3_left[m, n] * B_4_left[o, p] * I_4[j, l, n, p]
    # A_4_left /= A_4_left.norm()
    @einsum A_4_right[i, k, m, o] := B_1_right[i, j] * B_2_right[k, l] * B_3_right[m, n] * B_4_right[o, p] * I_4[j, l, n, p]
    # A_4_right /= A_4_right.norm()
    A_4_left = reshape(A_4_left, (1, 2, 2, 1, 2, 2))
    A_4_right = reshape(A_4_right, (2, 2, 1, 1, 2, 2))
    push!(row_tensor_mps, A_4_left)
    for col_id in 2: dims[3]-1
        j_vec = j_matrix[1, col_id, :]
        B_array = zeros(Complex, (5, 2, 2))
        j_vec = [j_vec[1], j_vec[2], j_vec[3], j_vec[5], j_vec[6]]
        for j in 1: 5 
            jj = j_vec[j]
            if jj == 1
                B_array[j, :, :] = B_pos
            else
                B_array[j, :, :] = B_neg
            end
        end
        B_1, B_2, B_3 = B_array[1, :, :], B_array[2, :, :], B_array[3, :, :]
        B_4, B_5 = B_array[4, :, :], B_array[5, :, :]
        @einsum A_5_i[i, k, m, o, q] := B_1[i, j] * B_2[k, l] * B_3[m, n] * B_4[o, p] * B_5[q, r] * I_5[j, l, n, p, r]
        # A_5_i = A_5_i / A_5_i.norm()
        A_5_i = reshape(A_5_i, (2, 2, 2, 1, 2, 2))
        push!(row_tensor_mps, A_5_i)
    end
    push!(row_tensor_mps, A_4_right)
    return row_tensor_mps
end

function up_row_tensor_mps_bdy_layer_create(j_matrix)
    row_tensor_mps = Any[]
    j_vec_left = [
        j_matrix[1, 1, 2], 
        j_matrix[1, 1, 3], 
        j_matrix[1, 1, 5],
    ]
    j_vec_right = [
        j_matrix[1, dims[3], 1], 
        j_matrix[1, dims[3], 2], 
        j_matrix[1, dims[3], 5], 
    ]
    B_array_left = zeros(Complex, (3, 2, 2))
    B_array_right = zeros(Complex, (3, 2, 2))
    for i in 1: 3
        jj = j_vec_left[i]
        kk = j_vec_right[i]
        if jj == 1
            B_array_left[i, :, :] = B_pos
        else
            B_array_left[i, :, :] = B_neg
        end
        if kk == 1
            B_array_right[i, :, :] = B_pos 
        else
            B_array_right[i, :, :] = B_neg 
        end
    end
    B_1_left, B_2_left = B_array_left[1, :, :], B_array_left[2, :, :]
    B_1_right, B_2_right = B_array_right[1, :, :], B_array_right[2, :, :]
    B_3_left, B_3_right = B_array_left[3, :, :], B_array_right[3, :, :]
    @einsum A_3_left[i, k, m] := B_1_left[i, j] * B_2_left[k, l] * B_3_left[m, n] * I_3[j, l, n]
    @einsum A_3_right[i, k, m] := B_1_right[i, j] * B_2_right[k, l] * B_3_right[m, n] * I_3[j, l, n]
    #A_3_left /= A_3_left.norm()
    #A_3_right /= A_3_right.norm()
    A_3_left = reshape(A_3_left, (1, 2, 2, 1, 2))
    A_3_right = reshape(A_3_right, (2, 2, 1, 1, 2))
    push!(row_tensor_mps, A_3_left)
    for col_id in 2: dims[3]-1
        j_vec = j_matrix[1, col_id, :]
        B_array = zeros(Complex, (4, 2, 2))
        j_vec = [j_vec[1], j_vec[2], j_vec[3], j_vec[5]]
        for j in 1: 4
            jj = j_vec[j]
            if jj == 1
                B_array[j, :, :] = B_pos
            else
                B_array[j, :, :] = B_neg
            end
        end
        B_1, B_2, B_3, B_4 = B_array[1, :, :], B_array[2, :, :], B_array[3, :, :], B_array[4, :, :]
        @einsum A_4_i[i, k, m, o] := B_1[i, j] * B_2[k, l] * B_3[m, n] * B_4[o, p] * I_4[j, l, n, p]
        # A_4_i = A_4_i / A_4_i.norm()
        A_4_i = reshape(A_4_i, (2, 2, 2, 1, 2))
        push!(row_tensor_mps, A_4_i)
    end
    push!(row_tensor_mps, A_3_right)
    return row_tensor_mps
end

function down_row_tensor_mps_bulk_layer_create(j_matrix)
    row_tensor_mps = Any[]
    j_vec_left = [
        j_matrix[dims[2], 1, 3], j_matrix[dims[2], 1, 4], 
        j_matrix[dims[2], 1, 5], j_matrix[dims[2], 1, 6],
    ]
    j_vec_right = [
        j_matrix[dims[2], dims[3], 1], j_matrix[dims[2], dims[3], 4], 
        j_matrix[dims[2], dims[3], 5], j_matrix[dims[2], dims[3], 6],
    ]
    B_array_left = zeros(Complex, (4, 2, 2))
    B_array_right = zeros(Complex, (4, 2, 2))
    for i in 1: 4
        jj = j_vec_left[i]
        kk = j_vec_right[i]
        if jj == 1
            B_array_left[i, :, :] = B_pos 
        else
            B_array_left[i, :, :] = B_neg 
        end
        if kk == 1
            B_array_right[i, :, :] = B_pos 
        else
            B_array_right[i, :, :] = B_neg 
        end
    end
    B_1_left, B_2_left = B_array_left[1, :, :], B_array_left[2, :, :]
    B_3_left, B_4_left = B_array_left[3, :, :], B_array_left[4, :, :]
    B_1_right, B_2_right = B_array_right[1, :, :], B_array_right[2, :, :]
    B_3_right, B_4_right = B_array_right[3, :, :], B_array_right[4, :, :]
    @einsum A_4_left[i, k, m, o] := B_1_left[i, j] * B_2_left[k, l] * B_3_left[m, n] * B_4_left[o, p] * I_4[j, l, n, p]
    # A_4_left /= A_4_left.norm()
    @einsum A_4_right[i, k, m, o] := B_1_right[i, j] * B_2_right[k, l] * B_3_right[m, n] * B_4_right[o, p] * I_4[j, l, n, p]
    # A_4_right /= A_4_right.norm()
    A_4_left = reshape(A_4_left, (1, 1, 2, 2, 2, 2))
    A_4_right = reshape(A_4_right, (2, 1, 1, 2, 2, 2))
    push!(row_tensor_mps, A_4_left)
    for col_id in 2: dims[3]-1
        j_vec = j_matrix[dims[2], col_id, :]
        j_vec = [j_vec[1], j_vec[3], j_vec[4], j_vec[5], j_vec[6]]
        B_array = zeros(Complex, (5, 2, 2)) 
        for j in 1: 5
            jj = j_vec[j]
            if jj == 1
                B_array[j, :, :] = B_pos
            else
                B_array[j, :, :] = B_neg
            end
        end 
        B_1, B_2 = B_array[1, :, :], B_array[2, :, :]
        B_3, B_4 = B_array[3, :, :], B_array[4, :, :]
        B_5 = B_array[5, :, :]
        @einsum A_5_i[i, k, m, o, q] := B_1[i, j] * B_2[k, l] * B_3[m, n] * B_4[o, p] * B_5[q, r] * I_5[j, l, n, p, r]
        # A_5_i = A_5_i / A_5_i.norm()
        A_5_i = reshape(A_5_i, (2, 1, 2, 2, 2, 2))
        push!(row_tensor_mps, A_5_i)
    end
    push!(row_tensor_mps, A_4_right)
    return row_tensor_mps
end

function down_row_tensor_mps_bdy_layer_create(j_matrix)
    row_tensor_mps = Any[]
    j_vec_left = [
        j_matrix[dims[2], 1, 3], 
        j_matrix[dims[2], 1, 4], 
        j_matrix[dims[2], 1, 5], 
    ]
    j_vec_right = [
        j_matrix[dims[2], dims[3], 1], 
        j_matrix[dims[2], dims[3], 4], 
        j_matrix[dims[2], dims[3], 5], 
    ]
    B_array_left = zeros(Complex, (3, 2, 2))
    B_array_right = zeros(Complex, (3, 2, 2))
    for i in 1: 3
        jj = j_vec_left[i]
        kk = j_vec_right[i]
        if jj == 1
            B_array_left[i, :, :] = B_pos 
        else
            B_array_left[i, :, :] = B_neg
        end
        if kk == 1
            B_array_right[i, :, :] = B_pos 
        else
            B_array_right[i, :, :] = B_neg
        end
    end
    B_1_left, B_2_left, B_3_left = B_array_left[1, :, :], B_array_left[2, :, :], B_array_left[3, :, :]
    B_1_right, B_2_right, B_3_right = B_array_right[1, :, :], B_array_right[2, :, :], B_array_right[3, :, :]
    @einsum A_3_left[i, k, m] := B_1_left[i, j] * B_2_left[k, l] * B_3_left[m, n] * I_3[j, l, n]
    # A_3_left /= A_3_left.norm()
    @einsum A_3_right[i, k, m] := B_1_right[i, j] * B_2_right[k, l] * B_3_right[m, n] * I_3[j, l, n]
    # A_3_right /= A_3_right.norm()
    A_3_left = reshape(A_3_left, (1, 1, 2, 2, 2))
    A_3_right = reshape(A_3_right, (2, 1, 1, 2, 2))
    push!(row_tensor_mps, A_3_left)
    for col_id in 2: dims[2]-1
        j_vec = j_matrix[dims[2], col_id, :]
        j_vec = [j_vec[1], j_vec[3], j_vec[4], j_vec[5]]
        B_array = zeros(Complex, (4, 2, 2))
        for j in 1: 4
            jj = j_vec[j]
            if jj == 1
                B_array[j, :, :] = B_pos
            else
                B_array[j, :, :] = B_neg
            end
        end
        B_1, B_2, B_3, B_4 = B_array[1, :, :], B_array[2, :, :], B_array[3, :, :], B_array[4, :, :]
        @einsum A_4_i[i, k, m, o] := B_1[i, j] * B_2[k, l] * B_3[m, n] * B_4[o, p] * I_4[j, l, n, p]
        # A_4_i = A_4_i / A_4_i.norm()
        A_4_i = reshape(A_4_i, (2, 1, 2, 2, 2))
        push!(row_tensor_mps, A_4_i)
    end
    push!(row_tensor_mps, A_3_right)
    return row_tensor_mps
end

function up_row_tensor_mps_create(j_matrix)
    row_tensor_mps = Any[]
    j_vec_left = [j_matrix[1, 1, 2], j_matrix[1, 1, 3]]
    j_vec_right = [j_matrix[1, dims[3], 1], j_matrix[1, dims[3], 2]]
    B_array_left = zeros(Complex, (2, 2, 2))
    B_array_right = zeros(Complex, (2, 2, 2))
    for i in 1:2
        jj = j_vec_left[i]
        kk = j_vec_right[i]
        if jj == 1
            B_array_left[i, :, :] = B_pos 
        else
            B_array_left[i, :, :] = B_neg
        end
        if kk == 1
            B_array_right[i, :, :] = B_pos 
        else 
            B_array_right[i, :, :] = B_neg
        end
    end
    B_1_left, B_2_left = B_array_left[1, :, :], B_array_left[2, :, :]
    B_1_right, B_2_right = B_array_right[1, :, :], B_array_right[2, :, :]
    @einsum A_2_left[i, k] := B_1_left[i, j] * B_2_left[k, l] * I_2[j, l]
    @einsum A_2_right[i, k] := B_1_right[i, j] * B_2_right[k, l] * I_2[j, l]
    # A_2_left /= A_2_left.norm()
    # A_2_right /= A_2_right.norm()
    A_2_left = reshape(A_2_left, (1, 2, 2, 1))
    A_2_right = reshape(A_2_right, (2, 2, 1, 1))
    push!(row_tensor_mps, A_2_left)
    for col_id in 2: dims[2]-1
        j_vec = j_matrix[1, col_id, :]
        B_array = zeros(Complex, (3, 2, 2))
        j_vec = [j_vec[1], j_vec[2], j_vec[3]]
        for i in 1: 3
            jj = j_vec[i]
            if jj == 1
                B_array[i, :, :] = B_pos
            else
                B_array[i, :, :] = B_neg
            end
        end
        B_1, B_2, B_3 = B_array[1, :, :], B_array[2, :, :], B_array[3, :, :]
        @einsum A_3_i[i, k, m] := B_1[i, j] * B_2[k, l] * B_3[m, n] * I_3[j, l, n]
        # A_3_i = A_3_i / A_3_i.norm()
        A_3_i = reshape(A_3_i, (2, 2, 2, 1))
        push!(row_tensor_mps, A_3_i)
    end
    push!(row_tensor_mps, A_2_right)
    return row_tensor_mps
end

function down_row_tensor_mps_create(j_matrix)
    row_tensor_mps = Any[]
    j_vec_left = [
        j_matrix[dims[2], 1, 3], 
        j_matrix[dims[2], 1, 3],
    ]
    j_vec_right = [
        j_matrix[dims[2], dims[3], 1], 
        j_matrix[dims[2], dims[3], 4],
    ]
    B_array_left = zeros(Complex, (2, 2, 2))
    B_array_right = zeros(Complex, (2, 2, 2))
    for i in 1: 2
        jj = j_vec_left[i]
        kk = j_vec_right[i]
        if jj == 1
            B_array_left[i] = B_pos 
        else
            B_array_left[i] = B_neg 
        end
        if kk == 1 
            B_array_right[i] = B_pos 
        else 
            B_array_right[i] = B_neg
        end
    end
    B_1_left, B_2_left = B_array_left[1, :, :], B_array_left[2, :, :]
    B_1_right, B_2_right = B_array_right[1, :, :], B_array_right[2, :, :]
    @einsum A_2_left[i, k] := B_1_left[i, j] * B_2_left[k, l] * I_2[j, l]
    @einsum A_2_right[i, k] := B_1_right[i, j] * B_2_right[k, l] * I_2[j, l]
    # A_2_left /= A_2_left.norm()
    # A_2_right /= A_2_right.norm()
    A_2_left = reshape(A_2_left, (1, 1, 2, 2))
    A_2_right = reshape(A_2_right, (2, 1, 1, 2))
    push!(row_tensor_mps, A_2_left)
    for col_id in 2: dims[2] - 1
        j_vec = j_matrix[dims[2], col_id, :]
        j_vec = [j_vec[1], j_vec[3], j_vec[4]]
        B_array = zeros(Complex, (3, 2, 2))
        for i in 1: 3
            jj = j_vec[i]
            if jj == 1
                B_array[i, :, :] = B_pos
            else
                B_array[i, :, :] = B_neg
            end
        end
        B_1, B_2, B_3 = B_array[1, :, :], B_array[2, :, :], B_array[3, :, :]
        @einsum A_3_i[i, k, m] := B_1[i, j] * B_2[k, l] * B_3[m, n] * I_3[j, l, n]
        # A_3_i = A_3_i / A_3_i.norm()
        A_3_i = reshape(A_3_i, (2, 1, 2, 2))
        !push(row_tensor_mps, A_3_i)
    end
    !push(row_tensor_mps, A_2_right)
    return row_tensor_mps
end

function tensor_mpo_bulk_layer_create(j_matrix)
    tensor_mpo_bulk_layers = Any[]
    up_row_tensor_bulk_layer_mps = up_row_tensor_mps_bulk_layer_create(j_matrix)
    push!(tensor_mpo_bulk_layers, up_row_tensor_bulk_layer_mps)
    for row_id in 2: dims[2]-1
        row_tensor_bulk_layer_mpo_i = row_tensor_mpo_bulk_layer_create(j_matrix, row_id)
        push!(tensor_mpo_bulk_layers, row_tensor_bulk_layer_mpo_i)
    end
    down_row_tensor_bulk_layer_mps = down_row_tensor_mps_bulk_layer_create(j_matrix)
    push!(tensor_mpo_bulk_layers, down_row_tensor_bulk_layer_mps)
    return tensor_mpo_bulk_layers
end

function tensor_mps_bdy_layer_create(j_matrix)
    tensor_mpo_bdy_layers = Any[]
    up_row_tensor_bdy_layer_mps = up_row_tensor_mps_bdy_layer_create(j_matrix)
    push!(tensor_mpo_bdy_layers, up_row_tensor_bdy_layer_mps)
    for row_id in 2: dims[2]-1
        row_tensor_bdy_layer_mpo_i = row_tensor_mpo_bdy_layer_create(j_matrix, row_id)
        push!(tensor_mpo_bdy_layers, row_tensor_bdy_layer_mpo_i)
    end
    down_row_tensor_bdy_layer_mps = down_row_tensor_mps_bdy_layer_create(j_matrix)
    push!(tensor_mpo_bdy_layers, down_row_tensor_bdy_layer_mps)
    return tensor_mpo_bdy_layers
end

function tensor_mpo_3d_create(j_matrix_array)
    tensor_bulk_list = Any[]
    num_layers = dims[1]
    j_matrix_up = j_matrix_array[1, :, :, :]
    tensor_mpo_bdy_layers_up = tensor_mps_bdy_layer_create(j_matrix_up)
    push!(tensor_bulk_list, tensor_mpo_bdy_layers_up)
    for i in 2: num_layers-1
        j_matrix = j_matrix_array[i, :, :, :]
        tensor_mpo_bulk_layer = tensor_mpo_bulk_layer_create(j_matrix)
        push!(tensor_bulk_list, tensor_mpo_bulk_layer)
    end
    j_matrix_down = j_matrix_array[num_layers, :, :, :]
    tensor_mpo_bdy_layers_down = tensor_mps_bdy_layer_create(j_matrix_down)
    push!(tensor_bulk_list, tensor_mpo_bdy_layers_down)
    return tensor_bulk_list
end

function row_tensor_mpo_create(j_matrix, row_id)
    row_tensor_mpo = Any[]
    j_vec_left = [
        j_matrix[row_id, 1, 2], 
        j_matrix[row_id, 1, 3], 
        j_matrix[row_id, 1, 4],
    ]
    j_vec_right = [
        j_matrix[row_id, dims[3], 1], 
        j_matrix[row_id, dims[3], 2], 
        j_matrix[row_id, dims[3], 4]
    ]
    B_array_left, B_array_right = zeros(Complex, (3, 2, 2)), zeros(Complex, (3, 2, 2)) 
    for i in 1: 3
        jj = j_vec_left[i]
        kk = j_vec_right[i]
        if jj == 1
            B_array_left[i, :, :] = B_pos
        else
            B_array_left[i, :, :] = B_neg
        end
        if kk == 1
            B_array_right[i, :, :] = B_pos
        else 
            B_array_right[i, :, :] = B_neg
        end
    end
    B_1_left, B_2_left = B_array_left[1, :, :], B_array_left[2, :, :]
    B_1_right, B_2_right = B_array_right[1, :, :], B_array_right[2, :, :]
    B_3_left, B_3_right = B_array_left[3, :, :], B_array_right[3, :, :]
    @einsum A_3_left[i, k, m] := B_1_left[i, j] * B_2_left[k, l] * B_3_left[m, n] * I_3[j, l, n]
    @einsum A_3_right[i, k, m] := B_1_right[i, j] * B_2_right[k, l] * B_3_right[m, n] * I_3[j, l, n]
    A_3_left = reshape(A_3_left, (1, 2, 2, 2))
    A_3_right = reshape(A_3_right, (2, 2, 1, 2))
    push!(row_tensor_mpo, A_3_left)
    for col_id in 2: dims[2]-1
        B_array = zeros(Complex, (4, 2, 2))
        j_vec = j_matrix[row_id, col_id, 1: 4]
        for i in 1: 4
            jj = j_vec[i]
            if jj == 1
                B_array[i, :, :] = B_pos
            else
                B_array[i, :, :] = B_neg
            end
        end 
        B_1, B_2 = B_array[1, :, :], B_array[2, :, :]
        B_3, B_4 = B_array[3, :, :], B_array[4, :, :]
        @einsum A_4_i[i, k, m, p] := B_1[i, j] * B_2[k, l] * B_3[m, n] * B_4[p, q] * I_4[j, l, n, q]
        # A_4_i = A_4_i / A_4_i.norm()
        push!(row_tensor_mpo, A_4_i)
    end
    push!(row_tensor_mpo, A_3_right)
    return row_tensor_mpo
end

function tensor_mpo_create(j_matrix)
    tensor_mpo = Any[]
    up_row_tensor_mps = up_row_tensor_mps_create(j_matrix)
    push!(tensor_mpo, up_row_tensor_mps)
    for row_id in 2: dims[1]-1
        row_tensor_mpo_i = row_tensor_mpo_create(j_matrix, row_id)
        push!(tensor_mpo, row_tensor_mpo_i)
    end
    down_row_tensor_mps = down_row_tensor_mps_create(j_matrix)
    push!(tensor_mpo, down_row_tensor_mps)
    return tensor_mpo
end

global I_2, I_3, I_4, I_up, I_down, I_3_up, I_3_down, I_2_up, I_2_down, I_5, I_6
global I_4_up, I_4_down, I_5_up, I_5_down, I_6_up, I_6_down

global B_pos, B_neg, spin_pos_up, spin_pos_down, spin_neg_up, spin_neg_down

global tensor_bulk_list
global tensor_mpo

function tensors_init(j_matrix_array)
    global I_2, I_3, I_4, I_up, I_down, I_3_up, I_3_down, I_2_up, I_2_down, I_5, I_6
    global I_4_up, I_4_down, I_5_up, I_5_down, I_6_up, I_6_down
    global B_pos, B_neg, spin_pos_up, spin_pos_down, spin_neg_up, spin_neg_down
    global tensor_mpo, tensor_bulk_list
    I_2, I_3, I_4, I_5, I_6, I_up, I_down, I_3_up, I_3_down, I_2_up, I_2_down, I_4_up, I_4_down, I_5_up, I_5_down, I_6_up, I_6_down = copy_tensor_create()
    B_pos, B_neg, spin_pos_up, spin_pos_down, spin_neg_up, spin_neg_down = node_tensor_create()
    #tensor_mpo = tensor_mpo_create(j_matrix_list[1])
    tensor_bulk_list = tensor_mpo_3d_create(j_matrix_array)
end


#global subgrid_1, subgrid_2, grid_3d
#global dims, beta, j_matrix_list, h

#dims_init = (3, 3, 3)
#beta_init, h_init = 1, 0

#parameters_init(dims_init, beta_init, h_init)
#tensors_init(j_matrix_array)