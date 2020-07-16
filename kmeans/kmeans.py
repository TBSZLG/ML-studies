import random
from typing import Iterable, Tuple, List


def kmeans_cluster(x: Iterable, k: int, steps: int) \
        -> Tuple[Iterable, Iterable]:
    """
    Cluster the data `x` using the k-Means algorithm.

    :param x: Datapoints to cluster.
    :param k: Number of classes/clusters.
    :param steps: Number of steps to perform optimization.
    :return: Class/cluster labels assigned to each data point along with the
        cluster centroids.
    """
    centroids = get_initial_centroids(x, k)

    i = 0
    while i <= steps:
        labels = [assign_to_centroid(point, centroids) for point in x]
        centroids = optimize_centroids(x, centroids, labels)
        i += 1

    labels = [assign_to_centroid(point, centroids) for point in x]

    return labels, centroids


def assign_to_centroid(point: Iterable, centroids: Iterable) -> int:
    """
    Assign the datapoint `point` to its closest centroid with respect to
    the euclidean distance.

    :param point: Datapoint to find closest centroid for.
    :param centroids: List of centroids.
    :return: Index of closest centroid to point.
    """
    distances = [vector_difference(point, cen) for cen in centroids]
    distances_abs = [list(map(abs, d)) for d in distances]
    distances_sum = list(map(sum, distances_abs))

    return distances_sum.index(min(distances_sum))


def optimize_centroids(x: Iterable, centroids: Iterable, labels: Iterable) \
        -> Iterable:
    """
    Optimize the centroids with respect to the chosen labels for the
    datapoints.

    The optimization step has a closed form solution as:

    .. math::

        \vec{\mu}_k = \frac{\sum_n r_{nk}\vec{x}_n}{\sum_n r_{nk}}

    , which is simply expressed as "Set mu_k equal to the mean of all Datapoints
    assigned to cluster k".

    :param x: Datapoints.
    :param centroids: Centroid coordinates.
    :param labels: Labels indicating datapoint centroid correspondences.
    """
    opt_centroids = []
    for c_idx, centroid in enumerate(centroids):
        member_idcs = [i for i in range(len(labels)) if labels[i] == c_idx]
        members = [x[i] for i in member_idcs]
        if len(members) > 0:
            opt_centroids.append(points_mean(members))

    return opt_centroids


def get_data_spread(x: Iterable) -> Tuple[float]:
    """
    Determine the maximum and minimum values in x- and y- coordinates of the
    passed datapoints.

    :param x: Datapoints to find spread for.
    :return: Tuple consisting of spread values in order [min_x, max_x, min_y,
        max_y].
    """
    xs = [point[0] for point in x]
    ys = [point[1] for point in x]

    return (int(min(xs)), int(max(xs)), int(min(ys)), int(max(ys)))


def get_initial_centroids(x: Iterable, k: int) -> List[Iterable]:
    """
    Determine `k` random starting centroids for data clustering.

    :param x: Datapoints to determine starting centroids for.
    :param k: Number of clusters.
    :return: List of centroids as list with [x, y] entries.
    """
    spread = get_data_spread(x)
    centroids = [[random.choice(range(spread[0], spread[1])),
                  random.choice(range(spread[2], spread[3]))]
                 for _ in range(k)]

    return centroids


def distortion_measure(x: Iterable, centroids: Iterable, indicators: Iterable) \
        -> int:
    """
    Calculate the _distortion measure_ as defined by [1].

    .. math::

        J = \sum_{n=1}^N \sum_{k=1}^K r_{nk} ||\vec{x}_n - \vec{\mu}_k||^2

    , where x are the datapoints and mu the cluster cenroids and r_nk the
    indicator variable specifying if datapoint x is assigned to cluster with
    centroid mu.

    .. [1] Bishop, Christopher M. Pattern recognition and machine learning.
        springer, 2006.

    :param x: Datapoints to cluster.
    :param centroids: Centroids of clusters.
    :param indicators: Indicator variables specifiying which of the K
        clusters the datapoint x is assigned to.
    """
    measure = 0
    for point_idx, datapoint in enumerate(x):
        centroid_idx = indicators[point_idx].index(1)
        centroid = centroids[centroid_idx]

        measure += abs(vector_difference(datapoint, centroid))**2


def vector_difference(a: Iterable, b: Iterable) -> float:
    """
    Calculate the difference between two vectors as a - b.

    Both vectors must be of same length/dimension!

    :param a: Vector a
    :param b: Vector b
    :return: Vector difference calculated as a-b.
    """
    assert len(a) == len(b), "Vectors must be of same dimension!"

    return [a[i] - b[i] for i in range(len(a))]


def points_mean(x: Iterable) -> List[int]:
    """
    Calculate the mean coordinate from a set of points.

    :param x: Datapoints.
    :return: Mean coordinate of points as [x, y] coordinates.
    """
    mean_x = sum(p[0] for p in x) / len(x)
    mean_y = sum(p[1] for p in x) / len(x)

    return [mean_x, mean_y]
