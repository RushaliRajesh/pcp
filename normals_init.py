import numpy as np 
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
from scipy.optimize import minimize
from scipy.linalg import lstsq
import os

def find_neighs(points, k=250):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm= 'kd_tree').fit(points)
    distances, indices = nbrs.kneighbors(points) 
    # print(points[indices])
    return distances[:,1:], indices[:,1:]

def pca_plane(points):
    # print(points.shape)
    pca = PCA(n_components=3)
    pca.fit(points)
    return pca.components_[2]/np.linalg.norm(pca.components_[2])

def process_pcd(pcd):
    _, indices = find_neighs(pcd, 250)
    neighs = pcd[indices]
    norm_esti_pca = Parallel(n_jobs=-1)(delayed(pca_plane)(point_neighs) for point_neighs in neighs)
    return np.array(norm_esti_pca)
    
if __name__ == "__main__":

    pcd_list = np.load("xyz_files.npy")
    normals_list = np.load("normals_files.npy")

    pcds=[]
    norms = []
    for pcd, normals in zip(pcd_list, normals_list):
        pcds.append(np.loadtxt(pcd))
        norms.append(np.loadtxt(normals))
    pcds = np.array(pcds)
    norms = np.array(norms)
    print(pcds.shape, norms.shape)

    '''neighborhood (normal init)'''
    init_norms =[]
    for pcd in pcds:
        init_norms.append(process_pcd(pcd))
            
    # np.save("init_normals.npy", np.array(init_norms))