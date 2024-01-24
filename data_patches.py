import numpy as np
import ckwrap
from normals_init import find_neighs
from joblib import Parallel, delayed
import os
import torch

# clean version

'''creating patches via clustering'''

def process_one_pcd(main_ind):
    all_pts = np.load("xyz_points.npy")
    all_norms = np.load("init_normals.npy")
    pts = all_pts[main_ind]
    norms = all_norms[main_ind]
    print(pts.shape, norms.shape)
    dists_ori, indices_ori = find_neighs(pts)
    neighs = pts[indices_ori]
    labels_per_row = np.apply_along_axis(lambda row: ckwrap.ckmeans(row, 5).labels, 1, dists_ori)

    mat = []
    print(f'using %d gpus: ', torch.cuda.device_count())
    pts = torch.from_numpy(pts)
    norms = torch.from_numpy(norms)
    indices_ori = torch.from_numpy(indices_ori)
    labels_per_row = torch.from_numpy(labels_per_row)

    for i,( ind, labels) in enumerate(zip(indices_ori, labels_per_row)):
        center = pts[i]
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
            sign = torch.divide(dot_pdts, np.abs(dot_pdts))
            area = torch.multiply(area, sign)
            sort_indices = torch.argsort(area)
            sorted_coords = coords[sort_indices]
            
            # return dot_pdts, area, crs_pdts, coords, sorted_coords, vec1, vec2, sign, closest, center, closest_ind
            
            mat.append(sorted_coords)
        mat = np.array(mat)
        return mat
        # np.save("patches/pcd_"+str(main_ind)+"/pnt_"+str(i)+".npy", mat)
        #     break
        # break

if __name__ == "__main__":
    print("hi")
    all_pts = np.load("xyz_points.npy")
    all_norms = np.load("init_normals.npy")
    # a = all_pts[0]
    # print(a.shape)
    # d, i = find_neighs(a)
    # print(d.shape, i.shape)
    mat= process_one_pcd(0)
    print(mat)