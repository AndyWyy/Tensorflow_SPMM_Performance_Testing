from queue import Queue
from threading import threading
from threading import currentThread
import tensorflow as tf 
import numpy as np
import scipy as scipy
import sys
import time 
import scanpy as sc
import os 


def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

def work(q, sparse_input_matrix):
    while True:
        if q.empty():
            return
        else:
            rhs = q.get()
            out_put = tf.sparse.sparse_dense_matmul(sparse_input_matrix,rhs)

if __name__ == '__main__':
    filename = sys.argv[1]
    input_matrix = scipy.io.mmread(filename)
    print(input_matrix)
    thread_num = 4
    q = Queue.Queue(thread_num)
    sparse_input_matrix = convert_sparse_matrix_to_sparse_tensor(input_matrix)
    m = sparse_input_matrix.shape[0]
    k = 32
    size = k / thread_num
    rhs = tf.random.uniform([m,k],0.0,2147483647.0,tf.float64)
    threads = []
    for i in range(thread_num):
        tmp_matrix = rhs.slice(rhs,[i*size,0],[size,m])
        q.put(tmp_matrix)
    start_time = time.time()
    for i in range (thread_num):
        t = threading.Thread(work,(q,sparse_input_matrix))
    for i in range (thread_num):
        threads[i].start()
    for i in range (thread_num):
        threads[i].join()
    end_time = time.time()
    print('spmm total time : ',round(end_time - start_time, 10),'secs')



    