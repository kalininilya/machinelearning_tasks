import numpy as np
import csv

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

# Read train data
tr_train = list(csv.reader(open('train.txt', 'rt', encoding="utf8"), delimiter='\t'))

buses = []
for x in tr_train:
    if not(x[3] in buses):
        buses.append(x[3])

# Create buses array
new_data = []
new_tr_train = []
for i in range(0,len(buses)):
    for j in tr_train:
        if (j[3] == buses[i]):
            new_tr_train.append(j)
    new_data.append(new_tr_train)
    new_tr_train = []

# Append points only in one direction
new_new_data = []
vector_len = 2
for i in new_data:
    for j in range(0, len(i)-vector_len):
        sumx = 0
        sumy = 0
        for k in range(0, vector_len):
            sumx += float(i[j+k][1])
            sumy += float(i[j+k][2])
        if not((sumx/vector_len >= float(i[j][1])) and (sumy/vector_len >=float(i[j][2]))):
            new_new_data.append(i[j])

tr_train_data = []
for x in new_new_data:
    temp_arr = [x[1],x[2]]
    tr_train_data.append(temp_arr)

X = np.array(tr_train_data)

# Compute DBSCAN
db = DBSCAN(eps=20, min_samples=290,n_jobs=-1).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
print(unique_labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
cluster_points = []
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]

    
    temp_points = [xy[:, 0], xy[:, 1]]
    cluster_points.append(temp_points)

    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=10)
    
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=1)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

# Compute cluster averages
counter = 0  
stops = []
for i in cluster_points:
    x = i[0] 
    y = i[1]
    new_x = []
    new_y = []
    for j in range(len(x)): 
         new_x.append(x[j].astype(np.float))

    for j in range(len(y)): 
         new_y.append(y[j].astype(np.float))
    point = [np.mean(new_x), np.mean(new_y)]
    stops.append(point)

stops = sorted(stops, key=sum, reverse=True)

print ('\n'.join(' '.join(str(cell) for cell in row) for row in stops))

# Writing to file
fh = open("final_output.txt", 'w+')
for item in stops:
  fh.write("%s\n" % item)
