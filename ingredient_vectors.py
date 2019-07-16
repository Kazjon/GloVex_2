import sys, gensim, io, logging
import cPickle as pickle
from sklearn.manifold import TSNE

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    input_file = sys.argv[1]
    stemmed_input = input_file.split(".")[0]
    fasttext_model = stemmed_input+ "_fasttext.model"
    tsne_model = stemmed_input+ "_tsne.model"
    recipes = []
    maxlen = 0
    with io.open(input_file, mode="r", encoding="utf-8") as rf:
        for line in rf:
            recipe = line.rstrip().split(" ")
            recipes.append(recipe)
            maxlen = max(len(recipe), maxlen)
    print maxlen

    ft_model = gensim.models.FastText(size=128, sg=0, window=maxlen, min_count=500, workers=8)
    ft_model.build_vocab(sentences=recipes)
    ft_model.train(sentences=recipes, epochs = 3, total_examples = len(recipes))

    ft_model.save(fasttext_model)
    print "Fasttext model saved."

    ft_vectors_embedded = TSNE(n_components=2).fit_transform(ft_model.wv.vectors)
    print "TSNE embedding generated."
    with open(tsne_model, "w") as pf:
        pickle.dump(ft_vectors_embedded,pf)
    print "TSNE embedding saved."

    print "BUTTER:",ft_model.most_similar(u"butter")
    print "BEEF:",ft_model.most_similar(u"beef")
    print "BROCCOLI:",ft_model.most_similar(u"broccoli")