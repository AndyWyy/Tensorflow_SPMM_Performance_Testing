import scanpy as sc
import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf 
import numpy as np
import scipy as scipy
import sys
import time 

def get_sparse_tensor(input_data):
    indices = list(zip(*input_data.nonzero()))
    return tf.SparseTensor(indices=indices, values=np.float64(input_data.data), dense_shape=input_data.get_shape())

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

if __name__ == '__main__':
    filename = sys.argv[1]
    print(tf.config.list_physical_devices())
    input_matrix = scipy.io.mmread(filename)
    print(input_matrix)
    #sparse_input_matrix = get_sparse_tensor(input_matrix)
    sparse_input_matrix = convert_sparse_matrix_to_sparse_tensor(input_matrix)

    m = sparse_input_matrix.shape[0]
    k = 1
    rhs = tf.random.uniform([m,k],0.0,2147483647.0,tf.float64)
    print(rhs)
    start_time = time.time()




    for i in range(100):
        out_put = tf.sparse.sparse_dense_matmul(sparse_input_matrix,rhs)
        # dense_sparse_input_matrix = tf.sparse_tensor_to_dense(sparse_input_matrix)
        # out_put = tf.matmul(dense_sparse_input_matrix,rhs,True)
    end_time = time.time()
    print('spmm total time : ',round(end_time - start_time, 10)/100,'secs')
    print(out_put)


    # print(input_matrix)

