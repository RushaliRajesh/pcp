import numpy as np
import ckwrap
from normals_init import find_neighs
from joblib import Parallel, delayed
import os
import torch
from torch.utils.data import Dataset

# clean version

'''creating patches via clustering'''

def process_one_pcd(main_ind, point_ind=0):
    all_pts = np.load("xyz_points.npy")
    all_norms = np.load("init_normals.npy")
    pts = all_pts[main_ind]
    norms = all_norms[main_ind]
    print(pts.shape, norms.shape)
    dists_ori, indices_ori = find_neighs(pts)
    neighs = pts[indices_ori]
    labels_per_row = np.apply_along_axis(lambda row: ckwrap.ckmeans(row, 5).labels, 1, dists_ori)

    print(f'using %d gpus: ', torch.cuda.device_count())
    #pts to gpu
    pts = torch.from_numpy(pts).cuda()
    norms = torch.from_numpy(norms).cuda()
    indices_ori = torch.from_numpy(indices_ori).cuda()
    labels_per_row = torch.from_numpy(labels_per_row).cuda()

    i = point_ind
    ind = indices_ori[i]
    labels = labels_per_row[i]
    # for i,( ind, labels) in enumerate(zip(indices_ori, labels_per_row)):
    center = pts[i]
    mat = []
    for clus in torch.unique(labels, dim=0):
        # ind_clus = ind[np.where(labels == clus)]
        ind_clus = ind[torch.where(labels==clus)]
        coords = pts[ind_clus]

        if(coords.shape[0] < 25):
            # 2 random integers
            for _ in range(25-(coords.shape[0])):
                a, b = torch.randint(1, coords.shape[0], (2,))
                a_n, b_n = torch.divide(a, (a+b)), torch.divide(b, (a+b))
                rnd_pt = torch.add(torch.multiply(a_n, (center-coords[a])), torch.multiply(b_n, (center-coords[b]))).unsqueeze(0)
                coords = torch.cat((coords, rnd_pt), dim=0)
        else:
            coords = coords[:25]
        
        # print("coords after adding random points: ", coords)

        
        vec2 = torch.subtract(center, coords)
        closest_ind = torch.argmin(torch.linalg.norm(vec2, dim=1))
        closest = coords[closest_ind]
        vec1 = torch.subtract(center, closest)
        vec2 = torch.cat((vec2[:closest_ind], vec2[closest_ind+1:]), dim=0)
        vec1 = torch.divide(vec1, torch.linalg.norm(vec1))
        vec2 = torch.divide(vec2, torch.linalg.norm(vec2, dim=1, keepdim=True))
        crs_pdts = torch.cross(vec1.expand_as(vec2), vec2)
        area = torch.linalg.norm(crs_pdts, dim=1)
        crs_pdts = torch.divide(crs_pdts, torch.linalg.norm(crs_pdts, dim=1, keepdim=True))
        dot_pdts = torch.tensordot(crs_pdts, norms[i].T, dims=1)
        sign = torch.divide(dot_pdts, torch.abs(dot_pdts))
        area = torch.multiply(area, sign)
        sort_indices = torch.argsort(area)
        sorted_coords = coords[sort_indices]
        
        # return dot_pdts, area, crs_pdts, coords, sorted_coords, vec1, vec2, sign, closest, center, closest_ind
        
        mat.append(sorted_coords)
        # print("device of mat: ", mat[-1].device)
    mat = np.array([tensor.cpu().numpy() for tensor in mat])

    return mat
        # np.save("patches/pcd_"+str(main_ind)+"/pnt_"+str(i)+".npy", mat)
        #     break
        # break
    
class Patches_dataset(Dataset):
    def __init__(self, pcds_path, norms_path, num_patches):
        self.pcds_path = pcds_path
        self.norms_path = norms_path
        self.num_patches = num_patches
        self.rnd_pcds = np.random.choice(380, (num_patches,), replace=True)
        self.rnd_ind = np.random.randint(0, 100000, (num_patches,))
        self.all_mats = []
        print("lens of rnd_pcds and rnd_ind: ", len(self.rnd_pcds), len(self.rnd_ind))
        for pid, point in zip(self.rnd_pcds, self.rnd_ind):
            mat = process_one_pcd(pid, point)
            self.all_mats.append(mat)                                                                                

    def __len__(self):
        return len(self.all_mats)
    
    def __getitem__(self, ind):
        return self.all_mats[ind]

if __name__ == "__main__":
    print("hi")
    all_pts = np.load("xyz_points.npy")
    all_norms = np.load("init_normals.npy")
    # mat= process_one_pcd(0)
    # mat = np.array(mat)
    # print(mat.shape)
    # print(type(mat))
    data1 = Patches_dataset("xyz_points.npy", "init_normals.npy", 10)
    for i in data1:
        print(i.shape)
        
