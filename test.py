from src.image_utils import compress_image
from src.metrics import * 
from pathlib import Path
from PIL import Image
import numpy as np

_ , centroids , new_img = compress_image(Path('data/input/bird.png'),4,20,1e-6,seed=2)

with Image.open(Path('data/input/bird.png')) as img:
    img = img.convert('RGB')
    img = np.array(img)
    print('MSE ',mse(new_img,img))
    print('Inertia ',inertia(img,new_img))
    print('PSNR ',PSNR(new_img,img))
    print('Compression Ratio ',compression_ratio(new_img,img,centroids))