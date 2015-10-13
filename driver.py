import cPickle
import math
import random
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np

from networks import RBFNetwork
from util.kmeans import get_centroids

NUM_BASES = [2, 4, 7, 11, 16]
ETAS = [0.01, 0.02]

def is_valid_partitioning(partitioning):
    for key, points in partitioning.iteritems():
        if len(points) == 0:
            return False
    return True

def plot_points(points):
    plt.plot([p[0] for p in points], [p[1] for p in points], 'ro')
    plt.axis([0, 1, -0.2, 1.2])

    t = np.arange(0., 1., 0.005)
    plt.plot(t, [0.5 + (0.4 * math.sin(2.0 * math.pi * x)) for x in t], 'r-')

def plot_function(f):
    t = np.arange(0., 1., 0.005)
    values = [f(val) for val in t]
    plt.plot(t, values, 'b--')

def mse(points, network):
    return sum(map(lambda p: (network.feed(p[0]) - p[1])**2, points)) / len(points)

def train(network, points, eta):
    for i in range(100):  # 100 epochs
        points = random.sample(points, len(points))  # Randomize the training order.
        for point in points:  # Train on every point
            network.train(eta, *point)

def run(points, eta, num_bases):
    # Run k-means to get the gaussian centers.
    partitions = get_centroids(points, num_bases)
    # Sometimes we find a way to not have each centroid with a point.
    while not is_valid_partitioning(partitions):
        partitions = get_centroids(points, num_bases)

    # Calculate the gaussian widths.
    centers = []
    widths = []
    for center, cluster in partitions.iteritems():
        centers.append(center)
        # if len(cluster) == 1:
        #     # We will need to calculate this after the other centers are calculated.
        #     widths.append(float('NaN'))
        # else:
        #     widths.append(sum([(p[0] - center)**2 for p in cluster]) / len(cluster))

    centers = sorted(centers)
    width = (centers[-1] - centers[0]) / float(2 * len(centers))**0.5
    widths = [width**2 for center in centers]

    # Handle any non-set widths
    # set_widths = [w for w in widths if not math.isnan(w)]
    # avg_width = sum(set_widths) / len(set_widths)
    # for idx in range(len(widths)):
    #     if math.isnan(widths[idx]):
    #         # This is a one data-point cluster, set its width to the average of the others.
    #         widths[idx] = avg_width

    # Generate the network.
    network = RBFNetwork(zip(centers, widths), eta)

    # Train the network.
    bases_rep = ', '.join(['(%.2f, %.4f)' % (c, w) for (c, w) in zip(centers, widths)])
    train(network, points, eta)

    # Print out the Mean-Squared-Error
    print 'Mean Squared Error: %.4f' % mse(points, network)

    plt.subplot(len(NUM_BASES), len(ETAS), NUM_BASES.index(num_bases) * 2 + ETAS.index(eta) + 1)
    plt.title('Approximation with %d bases and eta=%.2f' % (len(centers), eta))
    plt.xticks([0.2, 0.8])
    plt.yticks([0.0, 0.4, 0.8, 1.2])
    plot_points(points)
    plot_function(network.feed)

def main():
    # Load the data points.
    with open('data/points.pkl', 'rb') as points_file:
        points = cPickle.load(points_file)

    for eta in ETAS:
        for bases in NUM_BASES:
            run(points, eta, bases)

    plt.show()

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit('Received Keyboard Interrupt...Aborting.')
    except Exception:
        sys.exit(traceback.format_exc())
