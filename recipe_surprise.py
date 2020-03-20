from joblib import Parallel,delayed,parallel_backend
import sys,gensim,unicodecsv, re, io, json, os
import scipy.sparse
from itertools import izip
import numpy as np
from recipe_surprise_utils import corpus_surprise, load_clustering, cooc_chunk_clusters
import pandas as pd
from collections import Counter

NEIGHBOUR_EPSILON = 0.80

if __name__ == "__main__":
    recipe_file = sys.argv[1]
    gensim_model_import = sys.argv[2]
    cluster_file = sys.argv[3]
    cluster_labels_file = sys.argv[4]
    ingredient_freqs_file = sys.argv[5]
    cooc_output_file = sys.argv[6]
    surprise_output_file = sys.argv[7]
    most_surprising_ingredients_file = sys.argv[8]

    neighbourhood_cooc_output_file = cooc_output_file.split(".npz")[0] + "_neighbourhood" + "{:.2f}".format(NEIGHBOUR_EPSILON) + ".npz"
    neighbourhood_freqs_file = ingredient_freqs_file.split(".")[0] + "_neighbourhood" + "{:.2f}".format(NEIGHBOUR_EPSILON) + "_withclusters.csv"
    neighbourhood_surprise_file = surprise_output_file.split(".")[0] + "_neighbourhood" + "{:.2f}".format(NEIGHBOUR_EPSILON) + ".npz"

    ft_model = gensim.models.FastText.load(gensim_model_import)
    ft_vocab = ft_model.wv.index2word
    ft_vectors = ft_model.wv.vectors
    print "Loaded fasttext vectors, vocab size of",str(len(ft_vocab))+"."

    clustering, num_clusters, cluster_labels, cluster_name_to_index, superclusters, supercluster_names = load_clustering(cluster_file, cluster_labels_file)

    feature_freqs = {}
    for ing_name,ing_obj in ft_model.wv.vocab.iteritems():
        feature_freqs[ing_obj.index] = ing_obj.count

    num_ings = len(feature_freqs)
    print "Ingredient frequencies read in."

    with io.open(recipe_file,mode="r",encoding="utf-8") as rf:
        recipe_json = json.loads(rf.read())
    print "Parsed recipes read in."

    num_features = num_ings + num_clusters + len(supercluster_names)
    feature_coocs = scipy.sparse.dok_matrix((num_features, num_features))
    if not os.path.exists(cooc_output_file) or not os.path.exists(neighbourhood_cooc_output_file):
        feature_neighbourhoods = {}
        for fv in ft_vocab:
            feature_neighbourhoods[fv] = [ft_vocab[i[0]] for i in np.argwhere(ft_model.wv.most_similar(fv,topn=None)>=NEIGHBOUR_EPSILON)]
            feature_neighbourhoods[fv].remove(fv)
        n_cores = 6
        chunks_per_core = 20
        recipe_chunks = [[k, v] for k, v in recipe_json.items()]
        recipe_chunks = np.array_split(recipe_chunks, n_cores*chunks_per_core)
        print "Starting parallelised cooccurrence calculation."

        result = Parallel(n_jobs=n_cores, verbose=100)(delayed(cooc_chunk_clusters)(chunk, num_features, ft_model.wv.vocab, clustering, cluster_name_to_index, cluster_labels,
                                                                                    num_clusters,supercluster_names,feature_neighbourhoods) for chunk in recipe_chunks)
        feature_freq_subcounts, feature_cooc_submats, neighbourhood_feature_freq_subcounts, neighbourhood_feature_cooc_submats = zip(*result)
        cluster_feature_freqs = reduce(lambda x,y:x+y,feature_freq_subcounts)
        feature_freqs = dict(Counter(feature_freqs)+cluster_feature_freqs)
        feature_coocs = reduce(lambda x,y:x+y,feature_cooc_submats)
        neighbourhood_feature_freqs = reduce(lambda x,y:x+y,neighbourhood_feature_freq_subcounts)
        neighbourhood_feature_coocs = reduce(lambda x,y:x+y,neighbourhood_feature_cooc_submats)
        print "Finished calculating cluster and supercluster frequencies and co-occurrences."

        feature_coocs = feature_coocs.tocsr()
        feature_coocs += feature_coocs.transpose()
        with open(cooc_output_file,"wb") as cof:
            scipy.sparse.save_npz(cof, feature_coocs)
            print "Saved feature co-occurrence matrix file."

        with open(ingredient_freqs_file.split(".")[0]+"_withclusters.csv","wb") as iffu:
            writer = unicodecsv.writer(iffu)
            writer.writerows(sorted(feature_freqs.items(),key=lambda x:x[1],reverse=True))
            print "Saved updated feature frequencies file (with cluster features)."


        neighbourhood_feature_coocs = neighbourhood_feature_coocs.tocsr()
        neighbourhood_feature_coocs += neighbourhood_feature_coocs.transpose()
        with open(neighbourhood_cooc_output_file,"wb") as cof:
            scipy.sparse.save_npz(cof, neighbourhood_feature_coocs)
            print "Saved neighbourhood-based feature co-occurrence matrix file."

        with open(neighbourhood_freqs_file,"wb") as iffu:
            writer = unicodecsv.writer(iffu)
            writer.writerows(sorted(neighbourhood_feature_freqs.items(),key=lambda x:x[1],reverse=True))
            print "Saved neighbourhood-based feature frequencies file (with cluster features)."

    else:
        with open(cooc_output_file,"rb") as cof:
            feature_coocs = scipy.sparse.load_npz(cof)
            print "Loaded feature co-occurrence matrix file."

        with open(ingredient_freqs_file.split(".")[0]+"_withclusters.csv", "rb") as iffu:
            reader = unicodecsv.reader(iffu)
            feature_freqs = {}
            for row in reader:
                feature_freqs[int(row[0])] = int(float(row[1]))
            print "Loaded updated freature frequencies file."

        with open(neighbourhood_cooc_output_file, "rb") as cof:
            neighbourhood_feature_coocs = scipy.sparse.load_npz(cof)
            print "Loaded neighbourhood-based feature co-occurrence matrix file."

        with open(neighbourhood_freqs_file, "rb") as iffu:
            reader = unicodecsv.reader(iffu)
            neighbourhood_feature_freqs = {}
            for row in reader:
                neighbourhood_feature_freqs[int(row[0])] = int(float(row[1]))
            print "Loaded updated neighbourhood-based freature frequencies file."

    if not os.path.exists(surprise_output_file):
        ingredient_pair_surprises = corpus_surprise(feature_freqs, feature_coocs, ft_model.corpus_count, normalised_surprise=False)
        print ingredient_pair_surprises.row[:10]
        print ingredient_pair_surprises.col[:10]
        print ingredient_pair_surprises.data[:10]
        ingredient_pair_surprises = ingredient_pair_surprises.tocsr()
        with open(surprise_output_file, "wb") as sof:
            scipy.sparse.save_npz(sof, ingredient_pair_surprises)
            print "Saved ingredient surprise matrix file."
    else:
        with open(surprise_output_file,"rb") as sof:
            ingredient_pair_surprises = scipy.sparse.load_npz(sof)
            print "Loaded ingredient surprise matrix file."

    if not os.path.exists(neighbourhood_surprise_file):
        neighbourhood_pair_surprises = corpus_surprise(neighbourhood_feature_freqs, neighbourhood_feature_coocs, ft_model.corpus_count, normalised_surprise=False)
        print neighbourhood_pair_surprises.row[:10]
        print neighbourhood_pair_surprises.col[:10]
        print neighbourhood_pair_surprises.data[:10]
        neighbourhood_pair_surprises = neighbourhood_pair_surprises.tocsr()
        with open(neighbourhood_surprise_file, "wb") as sof:
            scipy.sparse.save_npz(sof, neighbourhood_pair_surprises)
            print "Saved neighbourhood-based ingredient surprise matrix file."
    else:
        with open(neighbourhood_surprise_file, "rb") as sof:
            neighbourhood_pair_surprises = scipy.sparse.load_npz(sof)
            print "Loaded neighbourhood-based ingredient surprise matrix file."


    #Use this version to limit to ingredients only (no clusters)
    #most_surprising_per_ing = [int(i) for i in ingredient_pair_surprises[:len(ft_vocab),:len(ft_vocab)].argmin(axis=1)]
    most_surprising_per_ing = [int(i) for i in ingredient_pair_surprises.argmin(axis=1)]

    feature_names = ft_vocab+[str(id) for id in range(num_clusters)]
    most_surprising_per_ing = sorted(zip(feature_names,
                                         (ft_vocab[i] if i < len(ft_vocab) else str(i) for i in most_surprising_per_ing),
                                         (ingredient_pair_surprises[k,i] for k,i in enumerate(most_surprising_per_ing))), key = lambda x:x[2], reverse=True)

    with open(most_surprising_ingredients_file, "wb") as msf:
        writer = unicodecsv.writer(msf)
        writer.writerows(most_surprising_per_ing)
