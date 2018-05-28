import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.datasets.samples_generator import make_blobs

from pandas.tools.plotting import parallel_coordinates

url = 'housing.csv'
cols =  ['id', 'area', 'room','floar','price']
data = pd.read_csv(url, names=cols)

#y = data['Class']          # Split off classifications
#print("-----------------classifications-----------")
#print(y)
X = data.ix[:, 'area':] # Split off features
print("-----------------features-----------")
print(X)

X_norm = (X - X.min())/(X.max() - X.min())
print("----------------- X_norm -----------")
print(X_norm)

pca = sklearnPCA(n_components=2) #2-dimensional PCA
transformed = pd.DataFrame(pca.fit_transform(X_norm))
print("----------------- transformed -----------")
print(transformed)

plt.scatter(transformed[0], transformed[1], c='red')
#plt.scatter(transformed[y==2][0], transformed[y==2][1], label='Class 2', c='blue')
#plt.scatter(transformed[y==3][0], transformed[y==3][1], label='Class 3', c='lightgreen')

plt.legend()
plt.show()
