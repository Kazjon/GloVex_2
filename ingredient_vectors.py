import sys, gensim, io, logging, csv
import cPickle as pickle
from sklearn.manifold import TSNE

generate_tsne = False

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    input_file = sys.argv[1]
    stemmed_input = input_file.split(".")[0]
    fasttext_model = stemmed_input+ "_fasttext.model"
    tsne_model = stemmed_input+ "_tsne.model"
    vocab_list = stemmed_input + "_vocablist.csv"
    recipes = []
    maxlen = 0
    with io.open(input_file, mode="r", encoding="utf-8") as rf:
        for line in rf:
            recipe = line.rstrip().split(" ")
            recipes.append(recipe)
            maxlen = max(len(recipe), maxlen)

    min_count = 20
    workers = 8

    ft_model = gensim.models.FastText(size=256, sg=0, window=maxlen, min_count=min_count, min_n = 3, max_n = 6, workers=workers)
    ft_model.build_vocab(sentences=recipes)
    ft_model.train(sentences=recipes, epochs = 10, total_examples = len(recipes))
    ft_model.save(fasttext_model)
    print "Fasttext model saved."

    if generate_tsne:
        ft_vectors_embedded = TSNE(n_components=10).fit_transform(ft_model.wv.vectors)
        print "TSNE embedding generated."
        with open(tsne_model, "w") as pf:
            pickle.dump(ft_vectors_embedded,pf)
        print "TSNE embedding saved."
    else:
        print "Skipping TSNE generation."

    print "BUTTER:",ft_model.most_similar(u"butter")
    print "BEEF:",ft_model.most_similar(u"beef")
    print "BROCCOLI:",ft_model.most_similar(u"broccoli")

    with open(vocab_list, "w") as vf:
        writer = csv.writer(vf)
        writer.writerows(sorted([(w[0],w[1].count) for w in ft_model.wv.vocab.iteritems()], key=lambda x:x[1], reverse=True))
    print "Vocab and frequencies saved."