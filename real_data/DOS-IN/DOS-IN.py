from numpy import array, ndarray, loadtxt, size, argsort, sort, sum, full, array, intersect1d, union1d, arange, argmax, bincount
from scipy.spatial.distance import pdist, squareform
from queue import Queue
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score, rand_score
import argparse
import time
from sklearn.neighbors import KDTree
import numpy as np
from sklearn.preprocessing import RobustScaler, Normalizer
import math

def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += math.pow(point2[i] - point1[i], 2)
    return math.sqrt(distance)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    vars(args)['algorithm'] = 'DOS-IN'

    # for convinience, -1 means unassigned, 0 means noise
    # DatasetName = args.dataset
    DatasetName_list = ['breastTissue.txt',
                        # 'digits.txt',
                        'divorce.txt', 'ecoli.txt', 'fertility_Diagnosis.txt', 'led7digit.txt', 'lenses.data', 'skewed.txt', 'Spiral.tsv', 'zoo.txt']
    # DatasetName_list = ['breastTissue.txt']
    for DatasetName in DatasetName_list:
        pathDataset = 'data/'+DatasetName

        unassigned = -1
        noise_label = 0

        # load data
        print(f"Dealing with {DatasetName}...")
        data: ndarray = loadtxt(pathDataset)
        dataCount = size(data, 0)
        label = data[:, -1]
        data = data[:, :-1]

        # 取对数(不可行）
        # data = np.log(data)

        # 鲁棒化
        # transfer = RobustScaler()
        # data = transfer.fit_transform(data)

        # Min-Max缩放
        maximums = np.max(data, axis=0)
        minimums = np.min(data, axis=0)
        for i in range(len(maximums)):
            for j in range(dataCount):
                data[j][i] = (data[j][i]-minimums[i])/(maximums[i]-minimums[i])

        # 方差缩放
        # mean_col = np.mean(data, axis=0)
        # var_col = np.var(data, axis=0)
        # for i in range(len(mean_col)):
        #     for j in range(dataCount):
        #         data[j][i] = (data[j][i]-mean_col[i])/var_col[i]

        # L2缩放
        # L2 = np.linalg.norm(data,axis=0)
        # for i in range(len(L2)):
        #     for j in range(dataCount):
        #         data[j][i] = data[j][i]/L2[i]


        # 正则化(结果很烂）
        # transfer = Normalizer()
        # data = transfer.fit_transform(data)

        # compute distance matrix
        distance = squareform(pdist(data, "euclidean"))
        distanceAsc = sort(distance)
        indexDistanceAsc = argsort(distance)
        F = sum(distanceAsc, axis=0)/dataCount
        # note that the first element = 0 because it refers to itself

        baseDelta = F[1] # 平均距离
        k_list = []
        NMI_list = []
        k_opt = 0
        NMI_opt = 0

        RI_list = []
        RI_opt = 0
        ARI_list = []
        ARI_opt = 0
        t = 0

        for k in arange(1, 11, 1):
            if DatasetName == 'digits.txt' and k > 2:
                break

            delta = k*baseDelta
            noise_ratio = 0.01
            print(
                f"baseDelta={baseDelta}, k={k}, delta={delta}, noise_ratio={0.01}")

            t1 = time.perf_counter()
            indexNeighbor = array([indexDistanceAsc[rowid][distanceAsc[rowid] < delta]
                                for rowid in range(dataCount)], dtype=object)
            numNeighbor = array([np.sum(distanceAsc[rowid] < delta)
                                for rowid in range(dataCount)])

            radius = full((dataCount, dataCount), 0.0)
            def cal_CN(array1, array2):
                set1 = set(array1.tolist())
                set2 = set(array2.tolist())
                return len(set1 & set2) / len(set1 | set2)

            for i in range(dataCount):
                for j in indexDistanceAsc[i][:numNeighbor[i]]:
                    if j == i:
                        continue
                    K = (numNeighbor[i] + numNeighbor[j]) //2
                    # intersectSet: ndarray = intersect1d(
                    #     indexDistanceAsc[i][:K], indexDistanceAsc[j][:K], assume_unique=True)
                    # unionSet: ndarray = union1d(
                    #     indexDistanceAsc[i][:K], indexDistanceAsc[j][:K])
                    # CN = intersectSet.size/unionSet.size
                    CN = cal_CN(indexDistanceAsc[i][:K], indexDistanceAsc[j][:K])
                    radius[i][j] = radius[j][i] = delta * CN

            # propagation and cluster assignment
            clusterStart = time.time()
            ii = 0
            typeFlag = 0
            q = Queue()
            cluster: ndarray = full(dataCount, unassigned)
            tmp_set: list = []
            tmp_set_len = 0
            reachable: ndarray = full((dataCount, dataCount), 0)
            while(ii < dataCount):
                q.put(ii)
                typeFlag = typeFlag+1
                cluster[ii] = typeFlag

                while(not q.empty()):
                    jj = q.get()
                    tmp_set.append(jj)
                    tmp_set_len = tmp_set_len + 1
                    for kk in range(ii+1, dataCount):
                        if(distance[kk][jj] < radius[kk][jj] and cluster[kk] == unassigned):
                            reachable[kk][jj] = typeFlag
                            cluster[kk] = typeFlag
                            q.put(kk)

                if(tmp_set_len < dataCount*noise_ratio):
                    typeFlag = typeFlag-1
                    for node in tmp_set:
                        cluster[node] = noise_label

                for nn in range(ii, dataCount):
                    if (cluster[nn] == unassigned):
                        ii = nn
                        break
                    elif (nn == dataCount-1):
                        ii = dataCount

                q.queue.clear()
                tmp_set.clear()
                tmp_set_len = 0

            # 计算初始簇中心
            cluster_uni = np.ndarray.flatten(cluster)
            cluster_mean = zeros_matrix = np.zeros((len(cluster_uni),size(data,1)))
            for i in cluster_uni:
                if i == 0 :
                    continue
                else:
                    data_tem = data[cluster == i][:]
                    cluster_mean[i-1][:] = data_tem.mean(axis=0)

            # 通过最近的簇中心合并噪声
            cluster_2 = full(dataCount, 0)
            for i in range(dataCount):
                if cluster[i] == noise_label:
                    min_dis = float('inf')
                    for j in cluster_uni:
                        dis = euclidean_distance(data[i][:], cluster_mean[j-1][:])
                        if dis < min_dis:
                            min_dis = dis
                            cluster_2[i] = j
            cluster = cluster + cluster_2

            # 通过最近点的聚类结果合并噪声
            # cluster_2 = full(dataCount, 0)
            # for i in range(dataCount):
            #     if cluster[i] == noise_label:
            #         for j in range(dataCount):
            #             if cluster[indexDistanceAsc[i][j]] != noise_label:
            #                 cluster_2[i] = cluster[indexDistanceAsc[i][j]]
            #                 break
            # cluster = cluster+cluster_2

            t2 = time.perf_counter()
            print(f"number of cluster = {typeFlag}, noise = {sum(cluster == noise_label)}")

            NMI = normalized_mutual_info_score(
                labels_pred=cluster, labels_true=label)
            AMI = adjusted_mutual_info_score(labels_pred=cluster, labels_true=label)
            ARI = adjusted_rand_score(labels_pred=cluster, labels_true=label)
            RI = rand_score(labels_pred=cluster, labels_true=label)
            print(f"NMI = {NMI}, AMI = {AMI}, ARI = {ARI}, RI = {RI}")
            k_list.append(k)
            NMI_list.append(NMI)
            RI_list.append(RI)
            if NMI > NMI_opt:
                NMI_opt = NMI
                k_opt = k
                t = t2-t1
            if RI > RI_opt:
                RI_opt = RI
            if ARI > ARI_opt:
                ARI_opt = ARI
            # if RI > RI_opt:
            #     RI_opt = RI
            #     k_opt = k
            #     t = t2-t1
            # if NMI > NMI_opt:
            #     NMI_opt = NMI
            # if ARI > ARI_opt:
            #     ARI_opt = ARI

            # if typeFlag==1 and sum(cluster == 1) == dataCount:
            #     break

        # plot figure
        dpi = 100
        fig, ax1 = plt.subplots(1)
        fig.suptitle(
            f'NMI_opt={NMI_opt:.4f}, k_opt={k_opt:.4f}, RI={RI_opt:.4f}, ARI={ARI_opt:.4f}, t={t:.6f}')
        ax1.scatter(k_list, NMI_list, s=3, c=NMI_list, cmap='rainbow')
        # plt.show()
        fig.savefig(
            fr'./fig-CC-MIN_MAX_NMI/{args.algorithm}_by1_{DatasetName[:-4]}_NMI.png', dpi=dpi)
