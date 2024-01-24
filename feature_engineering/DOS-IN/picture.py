import matplotlib.pyplot as plt
from numpy import array, ndarray, loadtxt, size, argsort, sort, sum, full, array, intersect1d, union1d, arange, argmax, bincount
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler


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

        # 标准化(方差缩放)
        # scaler = StandardScaler()
        # scaler.fit(data)
        # data = scaler.transform(data)

        # 鲁棒化
        # transfer = RobustScaler()
        # data = transfer.fit_transform(data)

        # Min-Max缩放
        maximums = np.max(data, axis=0)
        print(maximums)
        minimums = np.min(data, axis=0)
        print(minimums)
        print(len(maximums))
        for i in range(len(maximums)):
            for j in range(dataCount):
                data[j][i] = (data[j][i]-minimums[i])/(maximums[i]-minimums[i])

        # L2标准化
        # L2 = np.linalg.norm(data,axis=0)
        # for i in range(len(L2)):
        #     for j in range(dataCount):
        #         data[j][i] = data[j][i]/L2[i]

        # data = pd.DataFrame(data)
        # data.to_excel('./tem3.xlsx')
        y1 = data[:, 0]
        y2 = data[:, 1]
        y3 = data[:, 2]
        y4 = data[:, 3]
        y5 = data[:, 4]
        y6 = data[:, 5]
        y7 = data[:, 6]
        y8 = data[:, 7]
        y9 = data[:, 8]

        # Draw Plot
        ax = plt.figure(figsize=(10, 6), dpi=80)

        sns.kdeplot(y1, shade=True, color="orange", label="v1", alpha=.2)
        sns.kdeplot(y2, shade=True, color="green", label="v2", alpha=.2)
        sns.kdeplot(y3, shade=True, color="red", label="v3", alpha=.2)
        sns.kdeplot(y4, shade=True, color="blue", label="v4", alpha=.2)
        sns.kdeplot(y5, shade=True, color="purple", label="v5", alpha=.2)
        sns.kdeplot(y6, shade=True, color="yellow", label="v6", alpha=.2)
        sns.kdeplot(y7, shade=True, color="gray", label="v7", alpha=.2)
        sns.kdeplot(y8, shade=True, color="brown", label="v8", alpha=.2)
        sns.kdeplot(y9, shade=True, color="black", label="v9", alpha=.2)

        # Decoration
        ax.set_xlim = (50, 65)
        plt.legend(ncol=2, loc='best', fontsize=20)
        # plt.rcParams['font.size'] = 1  # 设置全局字体大小
        # plt.xlabel('(D)', fontsize=20)  # 设置X轴标签字体大小
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        plt.ylabel('Density', fontsize=20)  # 设置Y轴标签字体大小
        # plt.title('标题', fontsize=16)  # 设置图表标题字体大小
        plt.show()

