from math import log
from joblib import Parallel,delayed
import re
import scipy.sparse
from itertools import izip
import unicodecsv
import os
import itertools
from collections import Counter
import io

def pmi_scorer(worda_count, wordb_count, bigram_count, corpus_size,normalise=False, bits=False, alpha = 0.0):
    r"""Calculation NPMI score based on `"Normalized (Pointwise) Mutual Information in Colocation Extraction"
    by Gerlof Bouma <https://svn.spraakdata.gu.se/repos/gerlof/pub/www/Docs/npmi-pfd.pdf>`_.
    Parameters
    ----------
    worda_count : int
        Number of occurrences for first word.
    wordb_count : int
        Number of occurrences for second word.
    bigram_count : int
        Number of co-occurrences for phrase "worda_wordb".
    corpus_word_count : int
        Total number of words in the corpus.
    alpha : float
        Additive smoothing parameter (see https://en.wikipedia.org/wiki/Additive_smoothing)
    Returns
    -------
    float
        Score for given bi-gram, in the range -1 to 1 (if normalised) or inf to -inf if not.
    Notes
    -----
    Formula: :math:`\frac{ln(prop(word_a, word_b) / (prop(word_a)*prop(word_b)))}{ -ln(prop(word_a, word_b)}`,
    where :math:`prob(word) = \frac{word\_count}{corpus\_word\_count}`

    NOTE: ADAPTED/STOLEN FROM GENSIM: https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/phrases.py
    """
    pa = float(worda_count + alpha) / float(corpus_size + (2 * alpha))
    pb = float(wordb_count + alpha) / float(corpus_size + (2 * alpha))
    pab = float(bigram_count + alpha) / float(corpus_size + (2 * alpha))
    if bits:
        papb = pa * pb
        pmi = log(pab / papb,2)
        if normalise:
            pmi /= -log(pab,2)
    else:
        pmi = log(pab / (pa * pb))
        if normalise:
            pmi /= -log(pab)
    return pmi

def process_ing(ing):
    return re.sub(ur'\s+', u'_', ing)

def corpus_surprise(feature_frequencies, feature_coocs, total_features, normalised_surprise=False,bits=True,n_cpus=8):
    #index_pairs = combinations(feature_frequencies.keys(),2)
    #pairs_to_eval = ((feature_frequencies[a],feature_frequencies[b],feature_coocs[a,b]) for (a,b) in index_pairs if feature_coocs[a,b] > 0)
    feature_coocs = feature_coocs.tocoo()
    pairs_to_eval = ((a,b,feature_frequencies[a],feature_frequencies[b],c) for (a,b,c) in izip(feature_coocs.row,feature_coocs.col,feature_coocs.data))
    surprise_list = Parallel(n_jobs=n_cpus,batch_size=10000)(delayed(pmi_scorer)(na,nb,nab,total_features,normalise=normalised_surprise,bits=bits) for _,_,na,nb,nab in pairs_to_eval)
    return scipy.sparse.coo_matrix((surprise_list,(feature_coocs.row,feature_coocs.col)))

#Calculate the coocurrences on a chunk of the recipes, storing the result in a dok matrix that can later be summed.
def cooc_chunk(recipes,num_features,vocab):
    feature_coocs = scipy.sparse.dok_matrix((num_features, num_features),dtype=int)
    for id, recipe in recipes:
        for ing1 in recipe[u"ingredients"]:
            ing1 = process_ing(ing1)
            if ing1 in vocab:
                for ing2 in recipe[u"ingredients"]:
                    ing2 = process_ing(ing2)
                    if ing2 in vocab and vocab[ing1].index < vocab[ing2].index:
                        feature_coocs[vocab[ing1].index, vocab[ing2].index] += 1
    return feature_coocs

#Calculate the coocurrences on a chunk of the recipes, storing the result in a dok matrix that can later be summed.
def cooc_chunk_clusters(recipes,num_features,vocab, clustering, cluster_name_to_index, cluster_labels, num_clusters, supercluster_names, feature_neighbourhoods):
    feature_coocs = scipy.sparse.dok_matrix((num_features, num_features),dtype=int)
    feature_freqs = Counter()
    neighbourhood_feature_coocs = scipy.sparse.dok_matrix((len(vocab), len(vocab)),dtype=int)
    neighbourhood_feature_freqs = Counter()
    for id, recipe in recipes:
        base_ings = [process_ing(i) for i in recipe[u"ingredients"]]
        clustered_ings = append_clusters(base_ings, clustering, cluster_labels)
        clustered_ings = [[i] if isinstance(i, unicode) else i for i in clustered_ings]  # Just for this block we need all the features to be in lists, even the ones that aren't in clusters

        #Update the frequencies of each cluster and supercluster
        clusters_only = set([i[1] for i in clustered_ings if len(i) > 1])
        for cluster in clusters_only:
            if cluster in cluster_name_to_index:
                cluster_id = cluster_name_to_index[cluster]
            elif u"unlabelled_cluster_" in cluster:
                cluster_id = int(cluster.split("_")[2])
            feature_freqs[len(vocab)+cluster_id] += 1
        superclusters_only = set([i[2] for i in clustered_ings if len(i) > 2])
        for supercluster in superclusters_only:
            feature_freqs[len(vocab)+num_clusters+supercluster_names.index(supercluster)] += 1

        #Update the neighbourhood feature frequencies
        recipe_neighbourhood_freqs = Counter()
        for ing in base_ings:
            if ing in vocab:
                if vocab[ing].index not in recipe_neighbourhood_freqs:
                    recipe_neighbourhood_freqs[vocab[ing].index] +=1
                for neighbour in feature_neighbourhoods[ing]:
                    if vocab[neighbour].index not in recipe_neighbourhood_freqs:
                        recipe_neighbourhood_freqs[vocab[neighbour].index] += 1
        neighbourhood_feature_freqs += recipe_neighbourhood_freqs

        neighbourhood_cooc_pairs = set() #We need to be careful with logging neighbourhood coocs for cases where two neighbouring ingredients occur in the same recipe
        #Calculate co-occurrences for all three types of feature
        for (i1,i2) in itertools.permutations(clustered_ings,2):
            if i1[0] in vocab and i2[0] in vocab and vocab[i1[0]].index < vocab[i2[0]].index:
                i1_features = [vocab[i1[0]].index]
                i2_features = [vocab[i2[0]].index]
                #Unpack cluster and supercluster IDs if they're present.
                if len(i1) > 1:
                    i1_features.append(len(vocab)+clustering[i1[0]])
                    if len(i1) == 3:
                        i1_features.append(len(vocab)+num_clusters+supercluster_names.index(i1[2]))
                if len(i2) > 1:
                    i2_features.append(len(vocab)+clustering[i2[0]])
                    if len(i2) == 3:
                        i2_features.append(len(vocab)+num_clusters+supercluster_names.index(i2[2]))
                #Record the co-occurrences for all pairs, making sure to record in the upper triangle
                for (f1,f2) in itertools.product(i1_features,i2_features):
                    if f1<f2:
                        feature_coocs[f1,f2] += 1
                    else:
                        feature_coocs[f2,f1] += 1
                #Update neighbourhood co-occurrences
                for (nf1,nf2) in itertools.product([i1[0]]+feature_neighbourhoods[i1[0]],[i2[0]]+feature_neighbourhoods[i2[0]]):
                    if nf1 is not nf2:
                        nf1_i = vocab[nf1].index
                        nf2_i = vocab[nf2].index
                        neighbourhood_cooc_pairs.add(tuple(sorted([nf1_i,nf2_i])))
        for nf1,nf2 in neighbourhood_cooc_pairs:
            neighbourhood_feature_coocs[nf1,nf2] += 1
    return feature_freqs,feature_coocs,neighbourhood_feature_freqs,neighbourhood_feature_coocs

def load_clustering(cluster_file, cluster_labels_file):
    clustering = {}
    cluster_ids = set()
    with open(cluster_file,"rb") as cf:
        reader = unicodecsv.reader(cf)
        for row in reader:
            clustering[row[0]] = int(row[1])
            if int(row[1]) > -1:
                cluster_ids.add(int(row[1]))
    cluster_ids = sorted(list(cluster_ids))
    num_clusters = len(cluster_ids)
    print "Loaded HDBSCAN clustering."

    cluster_labels = {}
    cluster_name_to_index = {}
    superclusters = {}
    supercluster_names = set()
    removed = 0
    if os.path.exists(cluster_labels_file):
        with io.open(cluster_labels_file, mode="r",encoding="utf-8-sig") as clf:
            for line in clf:
                row = line.rstrip().split(u",")
                if len(row) >2 and not len(row[2]): #If there's a slot for a supercluster but it's empty.
                    row = row[:2]
                if row[1] == u"???":  #If this is a cluster we've identified as spurious (during the labelling process)
                    cluster_ids.remove(int(row[0]))
                    removed += 1
                    print "removing",row[0],"("+str(removed)+")"
                    for word,cluster in clustering.iteritems():
                        if cluster == int(row[0]):
                            clustering[word] = -1
                elif row[1] in [l[0] for l in cluster_labels.values()]: #If this is a cluster we want to merge (identified during the labelling process)
                    existing_id = cluster_labels.keys()[[l[0] for l in cluster_labels.values()].index(row[1])]
                    if int(row[0]) in cluster_ids:
                        cluster_ids.remove(int(row[0]))
                        removed += 1
                        print "removing",row[0],"("+str(removed)+")"
                    for word,cluster in clustering.iteritems():
                        if cluster == int(row[0]):
                            clustering[word] = existing_id
                else:
                    cluster_labels[int(row[0])] = [row[1]]
                    if len(row) > 2:
                        cluster_labels[int(row[0])].append(row[2])
                        superclusters[row[1]] = row[2]
                        supercluster_names.add(row[2])
        supercluster_names = sorted(list(supercluster_names))
        num_clusters = len(cluster_ids)
        #Go through all the clusters and re-index to skip over anything that was deleted.
        new_cluster_labels = {}
        for ix,id in enumerate(cluster_ids):
            if id in cluster_labels:
                new_cluster_labels[ix] = cluster_labels[id]
            if not id == ix:
                for name, cluster in clustering.iteritems():
                    if cluster == id:
                        clustering[name] = ix
        cluster_labels = new_cluster_labels
        cluster_name_to_index = {v[0]:k for k,v in cluster_labels.iteritems()}
        print "Loaded manual cluster labels."
    else:
        supercluster_names = []
        print "No manual cluster labels file found -- skipping labelling and superclustering."
    return clustering, num_clusters, cluster_labels, cluster_name_to_index, superclusters, supercluster_names

def append_clusters(ingredients, clustering, cluster_labels):
    clustered_ingredients = []
    for ing in ingredients:
        if ing in clustering and clustering[ing] > -1:  # if ing belongs to a cluster
            try:
                clustered_ingredients.append([ing]+cluster_labels[clustering[ing]])
            except KeyError:
                clustered_ingredients.append([ing, u"unlabelled_cluster_"+unicode(clustering[ing])])
            except:
                print "Hit an unexpected error on",ing,clustering[ing]
        else:
            clustered_ingredients.append(ing)
    return clustered_ingredients

def pairwise_surprise(i1n, i2n, i1i,i2i, observed_pairs, ingredient_pair_surprises, feature_freqs, feature_coocs, ft_model, bits=True, normalise=True, alpha=1):
    if (i1i, i2i) in observed_pairs:
        return (i1n, i2n, ingredient_pair_surprises[i1i, i2i], feature_freqs[i1i], feature_freqs[i2i], feature_coocs[i1i, i2i])
    else:
        if i1i > len(feature_freqs) or i2i > len(feature_freqs):
            pass
        i1i_f = feature_freqs[i1i]
        i2i_f = feature_freqs[i2i]
        i1i2_c = feature_coocs[i1i, i2i]
        surprise_estimate = pmi_scorer(i1i_f, i2i_f, i1i2_c, ft_model.corpus_count, bits=bits, normalise=normalise, alpha=alpha)
        return (i1n, i2n, surprise_estimate, feature_freqs[i1i], feature_freqs[i2i], feature_coocs[i1i, i2i])

def evaluate_recipe_surprise(ings, clustering, cluster_labels, ft_vocab, ft_model, num_clusters, supercluster_names, observed_pairs, ingredient_pair_surprises,
                             feature_freqs, feature_coocs, use_ings=False, use_clusters=True, use_superclusters=False, verbose=False):
    ings = append_clusters([process_ing(i) for i in ings], clustering, cluster_labels)
    surprise_list = []
    ings = [[i] if isinstance(i, str) or isinstance(i,unicode) else i for i in ings]  # Just for this block we need all the features to be in lists, even the ones that aren't in
    # clusters
    if verbose:
        for ing in ings:
            if ing[0] not in ft_vocab:
                print "not found:",ing[0]
    for (i1, i2) in itertools.combinations(ings, 2):
        if i1[0] in ft_vocab and i2[0] in ft_vocab:  # This is only here in the case that we're running with a model that does not contain all the ingredients in the mturk recipes, like with sample_100k
            i1_features = [(i1[0], ft_model.wv.vocab[i1[0]].index)]
            i2_features = [(i2[0], ft_model.wv.vocab[i2[0]].index)]
            # Unpack cluster and supercluster names if they're present, then calculate their surprises.
            if use_clusters and len(i1) > 1:
                if len(i2) == 1 or not i2[1] == i1[1]:  # Check to see that i1 and i2 aren't in the same cluster
                    i1_features.append((i1[1] + "_cluster", len(ft_vocab) + clustering[i1[0]]))
            if use_superclusters and len(i1) == 3:
                if len(i2) < 3 or not i2[2] == i1[2]:  # Check to see that i1 and i2 aren't in the same supercluster
                    i1_features.append((i1[2] + "_supercluster", len(ft_vocab) + num_clusters + supercluster_names.index(i1[2])))
            if use_clusters and len(i2) > 1:
                if len(i1) == 1 or not i1[1] == i2[1]:  # Check to see that i2 and i1 aren't in the same cluster
                    i2_features.append((i2[1] + "_cluster", len(ft_vocab) + clustering[i2[0]]))
            if use_superclusters and len(i2) == 3:
                if len(i1) < 3 or not i1[2] == i2[2]:  # Check to see that i1 and i2 aren't in the same supercluster
                    i2_features.append((i2[2] + "_supercluster", len(ft_vocab) + num_clusters + supercluster_names.index(i2[2])))
            if not use_ings:  # We need the base features to be in the list for calculating cluster cross-membership, so we always add them and then remove here if needed.
                i1_features = i1_features[1:] if len(i1_features) > 1 else i1_features
                i2_features = i2_features[1:] if len(i2_features) > 1 else i2_features

            for (f1, f2) in itertools.product(i1_features, i2_features):
                surprise_list.append(pairwise_surprise(f1[0], f2[0], f1[1], f2[1], observed_pairs, ingredient_pair_surprises, feature_freqs, feature_coocs, ft_model))
    surprise_list = sorted(list(set(surprise_list)), key=lambda x: x[2])
    return surprise_list

'''
def update_cluster_freqs_and_coocs(ingredient_occs, other_ingredient, ingredient_clustering, num_ings, num_clusters, feature_freqs, feature_coocs, cluster_labels, supercluster_names):
    ingredient_cluster_id = -1
    ing_supercluster_id = -1
    if ingredient_clustering > -1:  # if a belongs to a cluster
        ingredient_cluster_id = num_ings + ingredient_clustering # this is the ID in the cooc table and freq list (because clusters come after ingredients)
        if ingredient_cluster_id in feature_freqs:
            feature_freqs[ingredient_cluster_id] += ingredient_occs
        else:
            feature_freqs[ingredient_cluster_id] = ingredient_occs
        # Update the coocurrence of A's cluster with b (two-sided)
        # feature_coocs[a_cluster_id, b] += c
        feature_coocs[other_ingredient, ingredient_cluster_id] += ingredient_occs

        try:
            # Check to see if a is in a supercluster
            ing_supercluster_id = supercluster_names.index(cluster_labels[ingredient_clustering][1]) + num_ings + num_clusters
            # If we haven't errored out yet then update a's supercluster's coocs.
            if ing_supercluster_id in feature_freqs:
                feature_freqs[ing_supercluster_id] += ingredient_occs
            else:
                feature_freqs[ing_supercluster_id] = ingredient_occs
            # feature_coocs[a_supercluster_id, b] += c
            feature_coocs[other_ingredient, ing_supercluster_id] += ingredient_occs
        except (KeyError, ValueError, IndexError) as e:
            pass
    return ingredient_cluster_id, ing_supercluster_id
'''