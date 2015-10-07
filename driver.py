import cPickle
import math
import random
import sys
import traceback

from networks import RBFNetwork
from util.kmeans import get_centroids

NUM_BASES = [2, 4, 7, 11, 16]
ETAS = [0.01, 0.02]

def train(network, points):
    for i in range(100):  # 100 epochs
        random.shuffle(points)  # Randomize the training order.
        for point in points:  # Train on every point
            network.train(*point)

def run(points, eta, num_bases):
    # Run k-means to get the gaussian centers.
    partitions = get_centroids(points, num_bases)

    # Calculate the gaussian widths.
    centers = []
    widths = []
    for center, points in partitions.iteritems():
        centers.append(center)
        if len(points) == 1:
            # We will need to calculate this after the other centers are calculated.
            widths.append(float('NaN'))
        else:
            sigma_sq = sum([(p[0] - center)**2 for p in points]) / len(points)
            widths.append(sigma_sq)

    # Handle any non-set widths
    set_widths = [w for w in widths if not math.isnan(w)]
    avg_width = sum(set_widths) / len(set_widths)
    for idx in range(len(widths)):
        if math.isnan(widths[idx]):
            # This is a one data-point cluster, set its width to the average of the others.
            widths[idx] = avg_width

    # Generate the network.
    network = RBFNetwork(zip(centers, widths), eta)

    # Train the network.
    train(network, points)

def main():
    # Load the data points.
    with open('data/points.pkl', 'rb') as points_file:
        points = cPickle.load(points_file)

    for eta in ETAS:
        for bases in NUM_BASES:
            pass

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit('Received Keyboard Interrupt...Aborting.')
    except Exception:
        sys.exit(traceback.format_exc())
