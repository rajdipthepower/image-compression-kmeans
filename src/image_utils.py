from pathlib import Path
from PIL import Image
import numpy as np
from src.kmeans import *

def image_loading(path):
    p = Path(path)
    with Image.open(p) as img:
        img = img.convert('RGB')
        img = np.array(img)
        img = img/255.0
        return img

def prepare(path):
    img = image_loading(path)
    return img.shape[0],img.shape[1],img.reshape(-1,3) # shape:(number of images,features) where features are RGB values
# visualzing each pixel as a data point in RGB basis and collection of such pixels give an image

def compress_image(img_path,clusters,iterations,tol,seed):
    height,width,img = prepare(img_path) # Obtaining dimensions and image
    centroids,assign = clustering(iter=iterations,k=clusters,data=img,tol=tol,seed=seed) # eunning the algorithm
    new_img = centroids[assign] 
    new_img = new_img.reshape(height,width,-1)    # reconstructed image
    return assign,centroids,new_img # returning assignments,centroids and new image

def img_saving(new_image,save=False,save_name=None):
    if save:
        Image.fromarray((new_image*255.0).astype(np.uint8)).save(Path('data/output')/save_name)