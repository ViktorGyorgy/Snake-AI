import numpy as np
from collections import deque

snake = deque([np.array([0, 1]), np.array([0, 2]), np.array([0, 3])])

arr = np.array([0, 4])
print([all(np.equal(x, arr)) for x in snake])