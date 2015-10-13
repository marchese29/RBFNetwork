from copy import deepcopy
import math
import random

def is_same(p1, p2):
    assert p1.keys() == p2.keys()
    for num, points in p1.iteritems():
        p2points = p2[num]
        if set(points) != set(p2points):
            return False
    return True

def get_centroids(dataset, num_clusters):
    min_point = min(dataset, key=lambda x: x[0])
    max_point = max(dataset, key=lambda x: x[0])

    # Pick the centers as random points between the smallest and largest input values.
    centers = { i: random.uniform(min_point[0], max_point[0]) for i in range(num_clusters) }

    # Get the initial partitioning.
    partitioning = { i: [] for i in centers.keys() }
    for point in dataset:
        inp = point[0]
        current_min_dist = 100
        current_closest = None
        for cluster_num in centers.keys():
            if math.fabs(centers[cluster_num] - inp) < current_min_dist:
                current_min_dist = math.fabs(centers[cluster_num] - inp)
                current_closest = cluster_num
        partitioning[current_closest].append(point)
    for partition, points in partitioning.iteritems():
        if len(points) > 0:
            centers[partition] = sum([p[0] for p in points]) / len(points)
        else:
            centers[partition] = random.uniform(min_point[0], max_point[0])

    old_partitioning = { i: [] for i in centers.keys() }

    # Keep running until the partitioning does not change.
    while not is_same(partitioning, old_partitioning):
        old_partitioning = deepcopy(partitioning)
        partitioning = { i: [] for i in centers.keys() }

        # Re-partition the dataset.
        for point in dataset:
            inp = point[0]
            current_min_dist = 100
            current_closest = None
            for cluster_num in centers.keys():
                if math.fabs(centers[cluster_num] - inp) < current_min_dist:
                    current_min_dist = math.fabs(centers[cluster_num] - inp)
                    current_closest = cluster_num
            partitioning[current_closest].append(point)

        # Calculate the new centers
        for partition, points in partitioning.iteritems():
            if len(points) > 0:
                centers[partition] = sum([p[0] for p in points]) / len(points)
            else:
                centers[partition] = random.uniform(min_point[0], max_point[0])

    # Centroid location -> points in partition
    return { centers[num]: points for num, points in partitioning.iteritems() }
