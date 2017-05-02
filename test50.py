import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
arr = model.fit_transform(X)

plt.plot(arr[:, 0], arr[:, 1], '.')
plt.show()



