import numpy as np

def mse(new_img,old_img):
    old_img = old_img/255.0
    return np.mean((new_img-old_img)**2) # Using np.linalg.norm is less optimized than this

def PSNR(new_img,old_img):
    if np.max(new_img)<=1:
        return 10 * np.log10(1**2/mse(new_img,old_img))
    else:
        return 10 * np.log10(255.0**2/mse(new_img,old_img)) # Higher is better

def compression_ratio(new_img,old_img,centroids):
    old_bits = old_img.nbytes * 8 # nbytes give bytes occupied by each pixel(depending on dtype)
    centroids = np.clip(centroids*255,0,255).astype(old_img.dtype) # Converting centroids to same data type as old_image
    color_pallete = centroids.nbytes * 8 # Space occupied by color centers
    row,col,ch = new_img.shape
    new_bits = int(np.ceil(np.log2(len(centroids)))) * (row*col) + color_pallete # theoretical bits occupied by image + pallete
# Each pixel only refers to the cluster or centroid index(not assigned an actual RGB value which is done by pallete)
    return old_bits/new_bits # Higher,more compression

def inertia(old_img,new_img):
    dim = old_img.shape
    return dim[0]*dim[1]*dim[2]*mse(new_img,old_img)