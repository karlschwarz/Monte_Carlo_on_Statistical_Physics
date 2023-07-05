using Distributions
using LinearAlgebra

function j_matrix_2d(dims::Tuple, num_j::Int64)
    j_matrix = zeros(Int64, (dims[1], dims[2], num_j))
    for i in 1: dims[1]
        for j in 1: dims[2]
            p_binomial = Binomial(1, 1/2)
            sample_binomial = rand(p_binomial, num_j)
            j_matrix[i, j, :] = (-1) .^ sample_binomial
            #j_matrix[i, j, :] = ones(num_j)
            #if (j + 1) % 2 == 0
            #    j_matrix[i, j, :] = [1 -1 1 -1 1 1]
            #else
            #    j_matrix[i, j, :] = [-1 1 1 1 -1 -1]
            #end
        end
    end
    for i in 1: dims[1]
        for j in 1: dims[2]
            #j_matrix[i, j, 4] = j_matrix[(i - 1) % dims[1], j, 2]
            if i == 1
                j_matrix[i, j, 4] = j_matrix[dims[1], j, 2]
            else
                j_matrix[i, j, 4] = j_matrix[(i - 1), j, 2]
            end
            if j == 1
            #j_matrix[i, j, 1] = j_matrix[i, (j - 1) % dims[2], 3]
                j_matrix[i, j, 1] = j_matrix[i, dims[2], 3]
            else
                j_matrix[i, j, 1] = j_matrix[i, j - 1, 3]
            end
        end
    end
    return j_matrix
end

function j_matrix_2d_1(dims::Tuple, num_j::Int64)
    j_matrix = zeros(Int64, (dims[1], dims[2], num_j))
    for i in 1: dims[1]
        for j in 1: dims[2]
            p_binomial = Binomial(1, 1/2)
            sample_binomial = rand(p_binomial, num_j)
            j_matrix[i, j, :] = (-1) .^ sample_binomial
            #j_matrix[i, j, :] = ones(num_j)
            if (j + 1) % 2 == 0
                j_matrix[i, j, :] = [1 -1 1 -1 -1 1]
            else
                j_matrix[i, j, :] = [-1 1 -1 1 -1 -1]
            end
        end
    end
    for i in 1:dims[1]
        for j in 1: dims[2]
            #j_matrix[i, j, 4] = j_matrix[(i - 1) % dims[1], j, 2]
            if i == 1
                j_matrix[i, j, 4] = j_matrix[dims[1], j, 2]
            else
                j_matrix[i, j, 4] = j_matrix[(i - 1), j, 2]
            end
            if j == 1
            #j_matrix[i, j, 1] = j_matrix[i, (j - 1) % dims[2], 3]
                j_matrix[i, j, 1] = j_matrix[i, dims[2], 3]
            else
                j_matrix[i, j, 1] = j_matrix[i, j - 1, 3]
            end
        end
    end
    return j_matrix
end

function j_matrix_2d_2(dims::Tuple, num_j::Int64)
    j_matrix = zeros(Int64, (dims[1], dims[2], num_j))
    for i in 1: dims[1]
        for j in 1: dims[2]
            p_binomial = Binomial(1, 1/2)
            sample_binomial = rand(p_binomial, num_j)
            j_matrix[i, j, :] = (-1) .^ sample_binomial
            #j_matrix[i, j, :] = ones(num_j)
            if (j + 1) % 2 == 0
                j_matrix[i, j, :] = [-1 -1 1 -1 -1 1]
            else
                j_matrix[i, j, :] = [1 1 -1 -1 -1 1]
            end
        end
    end
    for i in 1:dims[1]
        for j in 1: dims[2]
            #j_matrix[i, j, 4] = j_matrix[(i - 1) % dims[1], j, 2]
            if i == 1
                j_matrix[i, j, 4] = j_matrix[dims[1], j, 2]
            else
                j_matrix[i, j, 4] = j_matrix[(i - 1), j, 2]
            end
            if j == 1
            #j_matrix[i, j, 1] = j_matrix[i, (j - 1) % dims[2], 3]
                j_matrix[i, j, 1] = j_matrix[i, dims[2], 3]
            else
                j_matrix[i, j, 1] = j_matrix[i, j - 1, 3]
            end
        end
    end
    return j_matrix
end

function j_matrix_2d_3(dims::Tuple, num_j::Int64)
    j_matrix = zeros(Int64, (dims[1], dims[2], num_j))
    for i in 1: dims[1]
        for j in 1: dims[2]
            p_binomial = Binomial(1, 1/2)
            sample_binomial = rand(p_binomial, num_j)
            j_matrix[i, j, :] = (-1) .^ sample_binomial
            #j_matrix[i, j, :] = ones(num_j)
            if (j + 1) % 2 == 0
                j_matrix[i, j, :] = [1 1 1 -1 -1 -1]
            else
                j_matrix[i, j, :] = [-1 1 -1 1 -1 -1]
            end
        end
    end
    for i in 1:dims[1]
        for j in 1: dims[2]
            #j_matrix[i, j, 4] = j_matrix[(i - 1) % dims[1], j, 2]
            if i == 1
                j_matrix[i, j, 4] = j_matrix[dims[1], j, 2]
            else
                j_matrix[i, j, 4] = j_matrix[(i - 1), j, 2]
            end
            if j == 1
            #j_matrix[i, j, 1] = j_matrix[i, (j - 1) % dims[2], 3]
                j_matrix[i, j, 1] = j_matrix[i, dims[2], 3]
            else
                j_matrix[i, j, 1] = j_matrix[i, j - 1, 3]
            end
        end
    end
    return j_matrix
end

function j_matrix_3d_1(dims::Tuple, num_j::Int64)
    global j_matrix_array
    j_matrix_array = zeros(Int64, (dims[1], dims[2], dims[3], num_j))
    num_layers = dims[1]
    dims_layer = (dims[2], dims[3])
    j_matrix = j_matrix_2d_1(dims_layer, num_j)
    j_matrix_array[1, :, :, :] = j_matrix
    j_matrix = j_matrix_2d_2(dims_layer, num_j)
    for i in 1: dims[2]
        for j in 1: dims[3]
            j_matrix[i, j, num_j-1] = j_matrix_array[1, i, j, num_j-1]
        end
    end
    j_matrix_array[2, :, :, :] = j_matrix
    for i in 3: num_layers
        j_matrix = j_matrix_2d_3(dims_layer, num_j)
        for j in 1: dims[2]
            for k in 1: dims[3]
                j_matrix[j, k, num_j-1] = j_matrix_array[i-1, j, k, num_j]
            end
        end
        j_matrix_array[i, :, :, :] = j_matrix 
    end
    return j_matrix_array
end

function j_matrix_3d(dims::Tuple, num_j::Int64)
    global j_matrix_array
    j_matrix_array = zeros(Int64, (dims[1], dims[2], dims[3], num_j))
    num_layers = dims[1]
    dims_layer = (dims[2], dims[3])
    j_matrix = j_matrix_2d(dims_layer, num_j)
    j_matrix_array[1, :, :, :] = j_matrix
    j_matrix = j_matrix_2d(dims_layer, num_j)
    for i in 1: dims[2]
        for j in 1: dims[3]
            j_matrix[i, j, num_j-1] = j_matrix_array[1, i, j, num_j-1]
        end
    end
    j_matrix_array[2, :, :, :] = j_matrix
    for i in 3: num_layers
        j_matrix = j_matrix_2d(dims_layer, num_j)
        for j in 1: dims[2]
            for k in 1: dims[3]
                j_matrix[j, k, num_j-1] = j_matrix_array[i-1, j, k, num_j]
            end
        end
        j_matrix_array[i, :, :, :] = j_matrix 
    end
    return j_matrix_array
end

