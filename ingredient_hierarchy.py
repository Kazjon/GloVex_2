import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gensim, sys, hdbscan
import cPickle as pickle
from collections import Counter
import unicodecsv
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

generate_tsne = True

gensim_model_import = sys.argv[1]
stemmed_input = gensim_model_import.split(".")[0]
tsne_model_import = sys.argv[2]
cluster_labels_file = sys.argv[3]
tsne_dims = int(sys.argv[4])
tsne_method = "barnes_hut"
if tsne_dims > 3:
    tsne_method = "exact"
tsne_metric = "euclidean"
if len(sys.argv) > 5:
    tsne_metric = sys.argv[5]

ft_model = gensim.models.FastText.load(gensim_model_import)
ft_vocab = ft_model.wv.index2word
ft_vectors = ft_model.wv.vectors
print "Loaded fasttext vectors, vocab size of",str(len(ft_vocab))+"."

if generate_tsne:
    ft_vectors_embedded = TSNE(n_components=tsne_dims, method=tsne_method, metric=tsne_metric).fit_transform(ft_model.wv.vectors)
    print "TSNE embedding generated."
    with open(tsne_model_import, "w") as pf:
        pickle.dump(ft_vectors_embedded, pf)
    print "TSNE embedding saved."
else:
    print "Skipping TSNE generation."

with open(tsne_model_import,"r") as pf:
    ft_vectors_embedded = pickle.load(pf)
print "Loaded TSNE embedding."

clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=2, core_dist_n_jobs=8, cluster_selection_method="eom")
print "Constructed clusterer."
clusterer.fit(ft_vectors_embedded)
print "Fit clusterer."
#clusterer.condensed_tree_.plot(cmap='viridis', colorbar=True)

num_clusters = len(set(clusterer.labels_))-1
num_outliers = sum(clusterer.labels_==-1)

print "Found",num_clusters,"clusters with",num_outliers,"outliers, for",num_clusters+num_outliers,"total features."
cluster_sizes = Counter(list(clusterer.labels_))
clusters_to_print = 5000
for cluster,cluster_size in cluster_sizes.most_common(clusters_to_print+1):
    if not cluster == -2:
        cluster_indices = np.where(clusterer.labels_==cluster)[0]
        print cluster, cluster_size
        print [ft_vocab[i] for i in cluster_indices]
color_palette = sns.color_palette('deep', len(set(clusterer.labels_)))
color_palette[-1] = (0.5,0.5,0.5)
cluster_colours = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in clusterer.labels_]
cluster_data = pd.DataFrame(ft_vectors_embedded,columns=["x","y","z"])
#sns.relplot(data = cluster_data)#, facecolors=cluster_colours)

if len(cluster_data.columns) == 3:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cluster_data["x"],cluster_data["y"],cluster_data["z"],c=cluster_colours,alpha=0.3)
else:
    plt.scatter(x=cluster_data["x"], y=cluster_data["y"], c=cluster_colours, alpha=0.3)
plt.show()

#clusterer.condensed_tree_.plot()
#plt.show()

with open(cluster_labels_file,"wb") as clf:
    writer = unicodecsv.writer(clf)
    for k,l in enumerate(clusterer.labels_):
        writer.writerow([ft_vocab[k],l])

'''
clusters_to_print = 1000
for distance in range(10):
    clusters = clusterer.single_linkage_tree_.get_clusters(cut_distance=distance,min_cluster_size=2)
    num_clusters = len(set(clusters))-1
    num_outliers = sum(clusters==-1)
    print "At distance", distance, "found",num_clusters,"clusters with",num_outliers,"outliers, for",num_clusters+num_outliers,"total features."
    cluster_sizes = Counter(list(clusters))
    for cluster, cluster_size in cluster_sizes.most_common(clusters_to_print + 1):
        if not cluster == -1:
            cluster_indices = np.where(clusters == cluster)[0]
            print cluster, cluster_size
            print [ft_vocab[i] for i in cluster_indices]
    color_palette = sns.color_palette('deep', len(set(clusters)))
    color_palette[-1] = (0.5, 0.5, 0.5)
    cluster_colours = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in clusters]
    cluster_data = pd.DataFrame(ft_vectors_embedded, columns=["x", "y"])
    plt.scatter(x=cluster_data["x"], y=cluster_data["y"], c=cluster_colours, alpha=0.3)
    plt.show()
    print "_____________________________"
'''

