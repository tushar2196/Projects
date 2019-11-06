import numpy as np
from kmodes.kprototypes import KPrototypes

#Data points with their publisher name,category score, category name, place name
syms = np.genfromtxt('iris.csv', dtype=str, delimiter=',')[1:]
X = np.genfromtxt('iris.csv', dtype=object, delimiter=',')[1:]
X[:, 0] = X[:, 0].astype(float)
kproto = KPrototypes(n_clusters=4, init='Huang', verbose=1)
clusters = kproto.fit_predict(X, categorical=[4])
# Print cluster centroids of the trained model.
print(kproto.cluster_centroids_)
# Print training statistics
clst0 , clst1 , clst2 , clst3 =[] , [] , [] , []
for i in range(len(clusters)):
	if clusters[i] == 0:
		lst0.append(syms[i])
	elif clusters[i] == 1:
		lst1.append(syms[i])
	elif clusters[i] == 2:
		lst2.append(syms[i])
	else:
		lst3.append(syms[i])	

with open('new.txt','w') as file:
	for g in lst2:
		lst = ''
		for l in g:
			lst += l+','
		file.write(lst[:-1]+'\n')	




# for s, c in zip(syms, clusters):
#     lst0.append("{},{}".format(s, c))   
# # Clustered result
# result = zip(syms, clusters)
# sortedR = sorted(result, key=lambda x: x[1])
# print(sortedR)