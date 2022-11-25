# 说明

spmm_test.py 测试tensorflow中稀疏矩阵乘法即tf.sparse.sparse_dense_matmul()的性能

spmm_test.py 多线程测试tensorflow中稀疏矩阵乘法tf.sparse.sparse_dense_matmul()的性能(采用简单的数据划分并行策略)

# 依赖

```shell
pip install tensorflow-gpu
pip install scipy
```

# 使用方法

```shell
#xxx.mtx是标准的稀疏矩阵文件
#GPU
python spmm_test.py xxx.mtx
#CPU
#把文件中第三行注释打开即可
python spmm_test.py xxx.mtx
#多线程 方法类似
python spmm_test_multi.py xxx.mtx
```

# 已知问题

GPU不支持大数据集读入，且多线程存在精度问题