import cPickle
from math import pi, sin
import random
import sys
import traceback

def main():
    points = []
    for i in range(75):
        x = random.uniform(0, 1)
        y = 0.5 + (0.4 * sin(2.0 * pi * x)) + random.uniform(-0.1, 0.1)
        points.append((x, y))

    with open('points.pkl', 'wb') as f:
        cPickle.dump(points, f)

    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit('Received Keyboard Interrupt...Aborting')
    except Exception:
        sys.exit(traceback.format_exc())
