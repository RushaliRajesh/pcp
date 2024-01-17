import numpy as np 
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
from scipy.optimize import minimize
from scipy.linalg import lstsq

path_norm = "pclouds/armadillo100k_ddist_minmax_layers.normals"
path_pcd = "pclouds/armadillo100k_ddist_minmax_layers.xyz"

normals = np.loadtxt(path_norm)
points = np.loadtxt(path_pcd)
# print(normals.shape, pcd.shape)    

'''visualization'''

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# pcd.normals = o3d.utility.Vector3dVector(normals)
# o3d.visualization.draw_geometries([pcd], point_show_normal=True)

'''neighborhood (normal init)'''

def find_neighs(points, k=250):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm= 'kd_tree').fit(points)
    distances, indices = nbrs.kneighbors(points) 
    return distances, indices

def pca_plane(points):
    # print(points.shape)
    pca = PCA(n_components=3)
    pca.fit(points)
    return pca.components_[2]/np.linalg.norm(pca.components_[2])

def scipy_ls(points):
    a = np.hstack((points[:, :2], np.ones(points.shape[0]).reshape(-1, 1)))
    b= points[:, 2]
    abc, residue, rank, s = lstsq(a, b) 
    return abc/np.linalg.norm(abc)

def plane_cost(params, points):
    a, b, c, d = params
    # Compute the signed distances from each point to the plane
    distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
    # Return the sum of squared distances as the cost
    return np.sum(distances**2)

def scipy_min(points):
    ini_guess= np.array([1.0,1.0,1.0,1.0])
    res = minimize(plane_cost, ini_guess, args=(points), method='Nelder-Mead', tol=1e-6)
    return res.x[:3]/np.linalg.norm(res.x[:3])

dists, indices = find_neighs(points, 250)
# print(pcd[indices].shape)
neighs = points[indices]

norm_esti_pca = Parallel(n_jobs=-1)(delayed(pca_plane)(point_neighs) for point_neighs in neighs)
norm_esti_ls = Parallel(n_jobs=-1)(delayed(scipy_ls)(point_neighs) for point_neighs in neighs)
norm_esti_min = Parallel(n_jobs=-1)(delayed(scipy_min)(point_neighs) for point_neighs in neighs)

norm_= normals/np.linalg.norm(normals[:300,:], axis=1)[:300,np.newaxis]
dot_pca = np.diag(np.dot(norm_esti_pca[:300,:] , norm_.T)).mean()
dot_ls = np.diag(np.dot(norm_esti_ls[:300,:] , norm_.T)).mean()
dot_min = np.diag(np.dot(norm_esti_min[:300,:] , norm_.T)).mean()

print(dot_pca, dot_ls, dot_min)
