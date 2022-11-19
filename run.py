import os

prefix = '/hzy/gyl/spmv/datasets'
dataset = ['/web-Google/web-Google.mtx','/flickr/flickr.mtx',
'/Freescale1/Freescale1.mtx','/cage14/cage14.mtx'
,'/soc-LiveJournal1/soc-LiveJournal1.mtx','/arabic-2005/arabic-2005.mtx']
for data in dataset:
    os.system("python spmm_test.py " + prefix + data)

