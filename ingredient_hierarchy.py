import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gensim, sys, hdbscan
import cPickle as pickle
from collections import Counter

gensim_model_import = sys.argv[1]
stemmed_input = gensim_model_import.split(".")[0]
tsne_model_import = sys.argv[2]

ft_model = gensim.models.FastText.load(gensim_model_import)
ft_vocab = ft_model.wv.index2word
ft_vectors = ft_model.wv.vectors
print "Loaded fasttext vectors, vocab size of",str(len(ft_vocab))+"."

with open(tsne_model_import,"r") as pf:
    ft_vectors_embedded = pickle.load(pf)
print "Loaded TSNE embedding."

clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=5, core_dist_n_jobs=8, cluster_selection_method="leaf")
print "Constructed clusterer."
clusterer.fit(ft_vectors)
print "Fit clusterer."
#clusterer.condensed_tree_.plot(cmap='viridis', colorbar=True)

print "Found",len(set(clusterer.labels_)),"clusters with",sum(clusterer.labels_==-1),"outliers."
cluster_sizes = Counter(list(clusterer.labels_))
clusters_to_print = 20
for cluster,cluster_size in cluster_sizes.most_common(clusters_to_print+1):
    if not cluster == -1:
        cluster_indices = np.where(clusterer.labels_==cluster)[0]
        print cluster, cluster_size
        print [ft_vocab[i] for i in cluster_indices]
color_palette = sns.color_palette('deep', len(set(clusterer.labels_)))
color_palette[-1] = (0.5,0.5,0.5)
cluster_colours = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in clusterer.labels_]
cluster_data = pd.DataFrame(ft_vectors_embedded,columns=["x","y"])
#sns.relplot(data = cluster_data)#, facecolors=cluster_colours)
plt.scatter(x=cluster_data["x"],y=cluster_data["y"],c=cluster_colours,alpha=0.3)
plt.show()



#clusterer.condensed_tree_.plot()
#plt.show()