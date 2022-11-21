from queue import Queue
from threading import threading
from threading import currentThread
mport tensorflow as tf 
import numpy as np
import scipy as scipy
import sys
import time 
import scanpy as sc
import os 



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
    for i in range(4):
        tmp_matrix = rhs.slice(rhs,[i*size,0],[size,m])
        q.put(tmp_matrix)
    