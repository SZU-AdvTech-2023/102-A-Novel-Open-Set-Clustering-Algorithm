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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from mpl_toolkits import mplot3d
from sklearn.cluster import KMeans

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
        data = pd.DataFrame(data)



        pd.options.display.max_columns = 10
        round(data.corr(), 2)
        sns.heatmap(round(data.corr(), 2), annot=True)
        plt.show()
        scaler = StandardScaler()
        scaler.fit(data)
        X = data
        # X = pd.DataFrame(X)
        # X.to_excel('./tem1.xlsx')
        model = PCA()
        model.fit(X)
        # X = model.transform(X)
        # X = pd.DataFrame(X)
        # a=X[0]
        # b=X[1]
        # c=X[2]
        # X=pd.concat([a,b,c],axis=1)
        #
        # X.to_excel('./tem2.xlsx')

        # 每个主成分能解释的方差
        model.explained_variance_
        # 每个主成分能解释的方差的百分比
        model.explained_variance_ratio_
        # 可视化
        plt.plot(model.explained_variance_ratio_, 'o-')
        plt.xlabel('Principal Component')
        plt.ylabel('Proportion of Variance Explained')
        plt.title('PVE')
        plt.show()
        plt.plot(model.explained_variance_ratio_.cumsum(), 'o-')
        plt.xlabel('Principal Component')
        plt.ylabel('Cumulative Proportion of Variance Explained')
        plt.axhline(0.9, color='k', linestyle='--', linewidth=1)
        plt.title('Cumulative PVE')
        plt.show()
        # 主成分核载矩阵
        model.components_

        columns = ['PC' + str(i) for i in range(1, 10)]

        pca_loadings = pd.DataFrame(model.components_, columns=data.columns, index=columns)
        k=round(pca_loadings, 2)
        print(k)
        # data1 = np.zeros((size(data,0), 3))
        # for i in range(4):
        #     for j in range(size(data,0)):
        #         data1[j][i] = X


        fig, ax = plt.subplots(2, 2)
        plt.subplots_adjust(hspace=1, wspace=0.5)
        for i in range(1, 4):
            ax = plt.subplot(2, 2, i)
            ax.plot(pca_loadings.T['PC' + str(i)], 'o-')
            ax.axhline(0, color='k', linestyle='--', linewidth=1)
            ax.set_xticks(range(9))
            ax.set_xticklabels(data.columns, rotation=30)
            ax.set_title('PCA Loadings for PC' + str(i))
        plt.show()
        # PCA Scores

        pca_scores = model.transform(X)
        pca_scores = pd.DataFrame(pca_scores, columns=columns)
        pca_scores.shape
        pca_scores.head()
        # 前两个主成分的可视化
        # visualize pca scores via biplot

        sns.scatterplot(x='PC1', y='PC2', data=pca_scores)
        plt.title('Biplot')
        plt.show()
        # Visualize pca scores via triplot

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pca_scores['PC1'], pca_scores['PC2'], pca_scores['PC3'], c='b')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.show()



        model = KMeans(n_clusters=3, random_state=1, n_init=20)
        model.fit(X)
        model.labels_

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pca_scores['PC1'], pca_scores['PC2'], pca_scores['PC3'],
                   c=model.labels_, cmap='rainbow')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.show()
        # print(X)
        # X=pd.DataFrame(X)
        # X.to_excel('./tem.xlsx')
