import numpy as np
from sklearn.cluster import k_means
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from kmeans import kmeans_cluster


if __name__ == '__main__':
    samples_a = np.random.multivariate_normal([3, 1], [[1, 0], [0, 1]], 100)
    labels_a = np.zeros((100))
    samples_b = np.random.multivariate_normal([6, 12], [[12, 3], [3, 1]], 25)
    labels_b = np.ones((25))
    samples_c = np.random.multivariate_normal([-13, 7], [[2, 14], [14, 3]],
                                              75)
    labels_c = np.ones((75)) * 2

    samples = np.vstack([samples_a, samples_b, samples_c])
    labels = np.hstack([labels_a, labels_b, labels_c])
    # np.random.shuffle(samples)

    pred_labels, pred_centroids = kmeans_cluster(samples, 3, 50)
    pred_centroids = np.asarray(pred_centroids)

    sklearn_cents, sklearn_labels, _ = k_means(samples, 3, init='random', n_init=1,
                                               max_iter=50)

    fig, axes = plt.subplots(3, 1)
    axes[0].scatter(samples[:, 0], samples[:, 1], c=labels)
    axes[0].set_aspect('equal')
    axes[0].set_title('Input samples')

    axes[1].scatter(samples[:, 0], samples[:, 1], c=pred_labels)
    axes[1].scatter(pred_centroids[:, 0], pred_centroids[:, 1], c='k', zorder=2)
    for cen in pred_centroids:
        ellipse = Ellipse(cen, 10, 10, facecolor='none', edgecolor='k')
        axes[1].add_patch(ellipse)
    axes[1].set_aspect('equal')
    axes[1].set_title('Self-implemented clustering')

    axes[2].scatter(samples[:, 0], samples[:, 1], c=sklearn_labels)
    axes[2].scatter(sklearn_cents[:, 0], sklearn_cents[:, 1], c='k', zorder=2)
    for cen in sklearn_cents:
        ellipse = Ellipse(cen, 10, 10, facecolor='none', edgecolor='k')
        axes[2].add_patch(ellipse)
    axes[2].set_aspect('equal')
    axes[2].set_title('Sklearn kmeans')

    plt.show()
