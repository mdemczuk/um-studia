import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering

data = pandas.read_csv('uczacy_3.csv')
test_data = pandas.read_csv('test.csv')

# Klastrowanie
kmeans = KMeans(4)
kmeans.fit(data)

identified_clusters = kmeans.fit_predict(data)
data_with_clusters = data.copy()
data_with_clusters['Clusters'] = identified_clusters

# -------------------------------------- kmeans test --------------------------------------
# wybieramy podzbior danych
y = test_data.iloc[:, 0:6]
test_clusters = kmeans.fit_predict(y)
test_data_with_clusters = test_data.copy()
test_data_with_clusters['Clusters'] = test_clusters
plt.scatter(data_with_clusters['sex'], data_with_clusters['team_size'], c=data_with_clusters['Clusters'], cmap='rainbow')
plt.show()
#

figure, axis = plt.subplots(1,2)
axis[0].scatter(data_with_clusters['smoke_room'], data_with_clusters['alcohol_choice'], c=data_with_clusters['Clusters'], cmap='rainbow')
axis[1].scatter(test_data_with_clusters['smoke_room'], test_data_with_clusters['alcohol_choice'], c=test_data_with_clusters['Clusters'], cmap='rainbow')
# plt.scatter(data_with_clusters['team_size'], data_with_clusters['alcohol_choice'], c=data_with_clusters['Clusters'], cmap='rainbow')
plt.show()
print('KMeans - all')
print(data_with_clusters.to_csv())
print('KMeans - test')
print(test_data_with_clusters.to_csv())

# -------------------------------------- AgglomerativeClustering --------------------------------------
aggCluster = AgglomerativeClustering(4)
aggCluster.fit(data)

identified_clusters2 = aggCluster.fit_predict(data)
data_with_clusters2 = data.copy()
data_with_clusters2['Clusters'] = identified_clusters2

print('AgglomerativeClustering - all')
print(data_with_clusters2.to_csv())
# -------------------------------------- AgglomerativeClustering test ---------------------------------
y = test_data.iloc[:, 0:6]
identified_clusters2 = aggCluster.fit_predict(y)
test_data_with_clusters2 = test_data.copy()
test_data_with_clusters2['Clusters'] = identified_clusters2

print('AgglomerativeClustering - test')
print(test_data_with_clusters2.to_csv())
