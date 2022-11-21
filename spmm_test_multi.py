from threading import currentThread
import queue
import threading
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

def work(q, sparse_input_matrix, sub_matrix_out):
    while True:
        if q.empty():
            return
        else:
            rhs = q.get()
            out_put = tf.sparse.sparse_dense_matmul(sparse_input_matrix,rhs)
            sub_matrix_out.put(out_put)

if __name__ == '__main__':
    filename = sys.argv[1]
    input_matrix = scipy.io.mmread(filename)
    print(input_matrix)
    thread_num = 4 # 线程数
    sub_matrix_out = queue.Queue(thread_num)
    q = queue.Queue(thread_num)
    sparse_input_matrix = convert_sparse_matrix_to_sparse_tensor(input_matrix)
    m = sparse_input_matrix.shape[0]
    k = 32 #列向量个数
    size = k / thread_num
    rhs = tf.random.uniform([m,k],0.0,100.0,tf.float64)
    threads = []
    print(rhs)
    #切分数据，分块
    for i in range(thread_num):
        tmp_matrix = tf.slice(rhs,[int(0),int(i*size)],[int(rhs.shape[0]),int(size)])
        print(tmp_matrix)
        q.put(tmp_matrix)
        print(i)
    start_time = time.time()
    for i in range(thread_num):
        t = threading.Thread(target=work,args=[q,sparse_input_matrix,sub_matrix_out])
        threads.append(t)
    for i in range(thread_num):
        threads[i].start()
    for i in range(thread_num):
        threads[i].join()
    out_put = sub_matrix_out.get()
    #合并矩阵
    for i in range(thread_num - 1):
        tmp = sub_matrix_out.get()
        out_put = tf.concat([out_put,tmp],1)
    print(out_put)
    print(tf.sparse.sparse_dense_matmul(sparse_input_matrix,rhs))
    end_time = time.time()
    print('spmm total time : ',round(end_time - start_time, 10),'secs')



    

