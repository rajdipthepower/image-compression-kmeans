import numpy as np

def compute_distance(centroid,data):
    # Need to subtract every data point from each centroid

    distance = np.zeros((data.shape[0],centroid.shape[0]))
    for i in range(centroid.shape[0]):
        distance[:,i] = np.sqrt(((data - centroid[i])**2).sum(axis = 1)) 
    # Much faster approach,numpy broadcasting creates large memory overhead,hence slower approach
    return distance # shape: (number of samples,centroids)

def centroid_assign(centroid,data):
    assigments = np.argmin(compute_distance(centroid,data),axis=1) # picking index of min distance of each sample with centroid
    return assigments # shape: (1D) (no of samples,)

def compute_centroid(centroid,data):
    assignments = centroid_assign(centroid,data)

    centers = []
    for i in range(len(centroid)):
        s = np.argwhere(assignments == i).flatten()

        if len(s) != 0 :

            centers.append(np.mean(data[s],axis=0))
        else:

            centers.append(centroid[i]) 

    return centers



def clustering(iter,k,data,tol,seed):
    rng = np.random.default_rng(seed)
    centers = data[rng.choice(data.shape[0],k,replace=False)]

    for i in range(iter):

        c = compute_centroid(centers,data)
        new_centers = np.array(c)  

        # print(f"Epoch {i+1}/{iter} Convergence : {np.sqrt(np.sum((new_centers - centers)**2))} Clusters : {p}")
        if np.sqrt(np.sum((new_centers - centers)**2)) < tol:
            return new_centers,centroid_assign(new_centers,data) # returning centroids and assignments
        else:
            centers = new_centers
    return centers,centroid_assign(centers,data)