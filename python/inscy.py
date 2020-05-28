from torch.utils.cpp_extension import load
import torch
import time

t0 = time.time()
print("Compiling our c++/cuda code, this usually takes 1-2 min. ")
inscy = load(name="inscy",
             sources=["inscy_map.cpp",
                      "src/utils/util.cu",
                      "src/utils/util_data.cu",
                      "src/utils/MergeUtil.cu",
                      "src/utils/RestrictUtils.cu",
                      "src/structures/ScyTreeNode.cu",
                      "src/structures/ScyTreeArray.cu",
                      "src/algorithms/clustering/ClusteringCpu.cu",
                      "src/algorithms/clustering/ClusteringGpu.cu",
                      "src/algorithms/clustering/ClusteringGpuStreams.cu",
                      "src/algorithms/inscy/InscyCompare.cu",
                      "src/algorithms/inscy/InscyNodeCpu.cu",
                      "src/algorithms/inscy/InscyCpuGpuMix.cu",
                      "src/algorithms/inscy/InscyCpuGpuMixClStream.cu",
                      "src/algorithms/inscy/InscyArrayGpu.cu",
                      "src/algorithms/inscy/InscyArrayGpuStream.cu"
                      ])
print("Finished compilation, took: %.4fs" % (time.time() - t0))


def normalize(x):
    min_x = x.min(0, keepdim=True)[0]
    max_x = x.max(0, keepdim=True)[0]
    # x_normed = x / x.max(0, keepdim=True)[0]
    x_normed = (x - min_x) / (max_x - min_x)
    return x_normed
    # min_v = torch.min(vector)
    # range_v = torch.max(vector) - min_v
    # if range_v > 0:
    #     normalised = (vector - min) / range_v
    # else:
    #     normalised = torch.zeros(vector.size())


def clean_up(subspaces, clusterings, min_size):
    number_of_different_subspaces = clusterings.size()[0]
    number_of_points = clusterings.size()[1]
    print(clusterings)
    empty_subspaces = [None] * number_of_different_subspaces
    for i in range(number_of_different_subspaces):
        bins = {}
        for j in range(number_of_points):
            cluster = clusterings[i, j].item()
            if cluster >= 0:
                if cluster in bins:
                    bins[cluster] += 1
                else:
                    bins[cluster] = 1
        # print(subspaces[i], bins)
        count = 0
        for bin in bins.keys():
            bin_size = bins[bin]
            if bin_size < min_size:
                clusterings[i][clusterings[i] == bin] = -1
            else:
                count += 1
        empty_subspaces[i] = count > 0
        bins = {}
        for j in range(number_of_points):
            cluster = clusterings[i, j].item()
            if cluster >= 0:
                if cluster in bins:
                    bins[cluster] += 1
                else:
                    bins[cluster] = 1
        print(subspaces[i], bins)

    return subspaces[empty_subspaces], clusterings[empty_subspaces]


def run_cpu(X, neighborhood_size, F, num_obj, min_size, r=1., number_of_cells=3):
    subspaces, clusterings = inscy.run_cpu(X, neighborhood_size, F, num_obj, min_size, r, number_of_cells)
    return clean_up(subspaces, clusterings, min_size)
    # return subspaces, clusterings


def run_cmp(X, neighborhood_size, F, num_obj, min_size, number_of_cells=3):
    return inscy.run_cmp(X, neighborhood_size, F, num_obj, min_size, number_of_cells)


def run_cpu_gpu_mix(X, neighborhood_size, F, num_obj, min_size, number_of_cells=3):
    subspaces, clusterings = inscy.run_cpu_gpu_mix(X, neighborhood_size, F, num_obj, min_size, number_of_cells)
    # return clean_up(subspaces, clusterings, min_size)
    return subspaces, clusterings


def run_cpu_gpu_mix_cl_steam(X, neighborhood_size, F, num_obj, min_size, number_of_cells=3):
    subspaces, clusterings = inscy.run_cpu_gpu_mix_cl_steam(X, neighborhood_size, F, num_obj, min_size, number_of_cells)
    # return clean_up(subspaces, clusterings, min_size)
    return subspaces, clusterings


def run_gpu(X, neighborhood_size, F, num_obj, min_size, number_of_cells=3):
    subspaces, clusterings = inscy.run_gpu(X, neighborhood_size, F, num_obj, min_size, number_of_cells)
    return clean_up(subspaces, clusterings, min_size)
    # return subspaces, clusterings


def run_gpu_stream(X, neighborhood_size, F, num_obj, min_size, number_of_cells=3):
    subspaces, clusterings = inscy.run_gpu_stream(X, neighborhood_size, F, num_obj, min_size, number_of_cells)
    # return clean_up(subspaces, clusterings, min_size)
    return subspaces, clusterings


def load_glove(n_max, d_max):
    return inscy.load_glove(n_max, d_max)
