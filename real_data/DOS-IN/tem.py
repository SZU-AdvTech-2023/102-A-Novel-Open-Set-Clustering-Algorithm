import matplotlib.pyplot as plt
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    vars(args)['algorithm'] = 'DOS-IN'

    # for convinience, -1 means unassigned, 0 means noise
    # DatasetName = args.dataset
    # DatasetName_list = ['breastTissue.txt', 'digits.txt', 'divorce.txt', 'ecoli.txt', 'fertility_Diagnosis.txt', 'led7digit.txt', 'lenses.data', 'skewed.txt', 'Spiral.tsv', 'zoo.txt']
    DatasetName_list = ['breastTissue.txt']
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
        for i in range(size(data,1)):
            plt.hist(data[i], bins=20, edgecolor='black')
            plt.show()