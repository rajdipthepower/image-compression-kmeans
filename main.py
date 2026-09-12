import numpy as np
from src.image_utils import compress_image
from pathlib import Path
import time
from PIL import Image
from src.metrics import *
import pandas as pd
import tracemalloc

K = [2,4,6,8,16,32,64]

# loading and keeping original image data
with Image.open(Path('data/input/bird.png')) as img:
    img = img.convert('RGB')
    img = np.array(img)

m = {'time':[],'mse':[],'psnr':[],'compression_ratio':[],'cluster/seed':[],'inertia':[],'memory':[]}
for k in K:
    print(f'-----------------------------------Starting Cluster {k}-------------------------------------------------------------')
    for i in range(4):
        t1 = time.time()
        tracemalloc.start()
        _,centroids,new_img = compress_image(Path('data/input/bird.png'),k,20,1e-5,seed=i)
        current , peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        m['time'].append(time.time()-t1)
        m['memory'].append(peak/1024**2)
        m['mse'].append(mse(new_img,img))
        m['psnr'].append(PSNR(new_img,img))
        m['cluster/seed'].append((k,i))
        m['compression_ratio'].append(compression_ratio(new_img,img,centroids))
        m['inertia'].append(inertia(img,new_img))
        print(f'Time = {round(m["time"][-1],2)}s Peak Memory Usage = {round(m["memory"][-1],2)} MB (Cluster,Seed) = {m["cluster/seed"][-1]} MSE = {m["mse"][-1].round(4)} PSNR = {m["psnr"][-1].round(4)} Inertia = {m["inertia"][-1].round(4)} Compression Ratio = {round(m["compression_ratio"][-1],2)}')
    print(f'---------------------------------------------------Cluster {k} done !! ---------------------------------------------')
df = pd.DataFrame(m)
df.to_csv('data/output/experiment_results.csv',index=False)
print('experiments_results.csv saved successfully at data/output folder !!!')