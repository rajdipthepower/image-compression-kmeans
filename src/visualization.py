import pandas as pd
import ast
import matplotlib.pyplot as plt
from src.image_utils import compress_image,img_saving
from pathlib import Path
from PIL import Image
import numpy as np

df = pd.read_csv('data/output/experiment_results.csv')  
df[['cluster', 'seed']] = pd.DataFrame(df['cluster/seed'].apply(ast.literal_eval).tolist(), index=df.index)
# saving the dataframe in csv causes column of cluster/seed to become strings, use ast.literal_evals to convert back to tuples
df = df.drop(columns=['cluster/seed'])
df_avg = df.groupby('cluster').mean(numeric_only=True).drop(columns=['seed'])


def graph(ylabel,title,ls,color):
    plt.figure(figsize=(8,6))
    plt.plot(df['cluster'],df['mse'],ls,marker='o',color = color)
    plt.xlabel('Clusters (K)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.savefig(f'data/output/{title}.png')
    plt.show()
graph('Mean Squared Error','Mean Squared Error vs Clusters','-','red')
graph('Time(s)','Time taken vs Clusters','--','green')
graph('PSNR','PSNR vs Clusters','--','blue')
graph('Compression Ratio','Compression Ratio vs Clusters','-','brown')
graph('Inertia','Inertia vs Clusters','-','black')
graph('Peak Memory Usage (MB)','Memory Usage vs Number of Clusters','--','yellow')

K=[2,4,6,8,16,32,64]
idx = df.groupby('cluster')['inertia'].idxmin() # getting the seed for a cluster having minimum inertia
fig,axes = plt.subplots(2,4,figsize=(11,7))
axes = axes.flatten()
with Image.open(Path('data/input/bird.png')) as img:
    axes[0].imshow(np.array(img.convert('RGB')))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
for index,i in enumerate(K):
    _,_,new_img = compress_image(Path('data/input/bird.png'),i,50,1e-5,idx.loc[i])
    img_saving(new_img,save=True,save_name=f'Image with clusters {i}.png')
    axes[index+1].imshow((new_img*255.0).astype(np.uint8))
    axes[index+1].set_title(f'Image using K={i}')
    axes[index+1].axis('off')
plt.suptitle("Reconstructing an image using Color Quantization via Kmeans")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('data/output/final_images.png',dpi = 300)
plt.show()
