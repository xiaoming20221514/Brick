import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
import pandas as pd
from sklearn.metrics import rand_score
from sklearn.mixture import GaussianMixture


def show_scatter(sub_fig, x, y, c, x_label='x', y_label='y', sub_title='sub_title'):
    sub_fig.scatter(x, y, c=c, cmap='Paired')
    sub_fig.set_title(sub_title)
    sub_fig.set_xlabel(x_label)
    sub_fig.set_ylabel(y_label)
    # sub_fig.legend()


if __name__ == '__main__':
    ######1. 加载数据
    data = pd.read_csv('data.csv')
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values

    ######2. 应用KMeans、GaussianMixture、DBSCAN、AgglomerativeClustering进行聚类，并分别调优，
    ######******tips：应在报告中说明每种模型需要对哪些参数进行重点调优，最终的参数值是什么、对应的模型打分值是什么*******######
    # 使用KMeans算法进行聚类
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X)
    # 评分
    ri_kmeans = rand_score(Y, kmeans.labels_)
    print('kmeasn打分:',ri_kmeans)


    # 3. 可视化结果
    # 3.1 将窗口分成4块
    _, axs = plt.subplots(2, 2, figsize=(12, 6))
    # 3.2 左上角子图显示KMeans结果
    show_scatter(axs[0][0],X[:, 0],X[:, 1],kmeans.labels_, 'x', 'y','kmeans cluster result')
    # 3.3-3.5 其他模型放在其他子图中显示

    plt.show()

