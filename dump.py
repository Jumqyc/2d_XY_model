from XY import XY
from multiprocessing import Pool
import numpy as np
import pickle as pkl
import os
import matplotlib.pyplot as plt
from itertools import product

dumping_area = './data'
os.makedirs(dumping_area, exist_ok=True)

def run_model(L:int,t:float):
    round(t,4)

    model = XY(t,L)
    model.run(20)
    repeat = 40
    num_test = 1024
    m = np.zeros((num_test,repeat))
    e  = np.zeros((num_test,repeat))
    h  = np.zeros((num_test,repeat))
    for i in range(repeat):
        model.run(100)
        m[:,i] = model.get_m()
        e[:,i] = model.get_e()
        h[:,i] = model.get_h()  
    h += e / 2
    h*=-1
    
    folder = f'{dumping_area}/L={L}'
    os.makedirs(folder,exist_ok=True)
    d = {
        't':t,
        'L':L,
        'm':m,
        'e':e,
        'h':h,
        'spin':model.get_spin()
    }
    with open(f'{folder}/T={t}.pkl','wb') as writter:
        pkl.dump(d,writter)

temp = np.round(np.arange(0.025,1.5,0.025),4)
size = [8,16,32,48,64,96,128,196,256][::-1]


tasks = product(size,temp)
with Pool() as p:
    p.starmap(run_model,tasks)