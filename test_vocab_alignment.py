import sys,gensim,unicodecsv, re, io, collections

gensim_model_import = sys.argv[1]
parsed_ingredient_freqs = sys.argv[2]
recipes_file = sys.argv[3]

ft_model = gensim.models.FastText.load(gensim_model_import)
ft_vocab = ft_model.wv.index2word
valid_ingredients = set(ft_vocab)

minimum_threshold = 0

total_found = 0
total_failed = 0

with open(parsed_ingredient_freqs, "rb") as iffu:
    reader = unicodecsv.reader(iffu)
    feature_freqs = {}
    for row in reader:
        freq = int(float(row[1]))
        feature_freqs[row[0]] = freq
        if freq > minimum_threshold:
            if not re.sub(ur'\s+', u'_', row[0]) in valid_ingredients:
                print "Not Found:",row[0],"("+row[1]+" occurrences)."
                total_failed += 1
            else:
                total_found += 1

print "Found",total_found,"out of",len(ft_vocab),"ingredients."
print "Failed",total_failed,"out of",len(feature_freqs.keys()),"ingredients."

recipes = []
completeness_counter = collections.Counter()
with io.open(recipes_file, mode="r", encoding="utf-8") as rf:
    for line in rf:
        recipe = line.rstrip().split(" ")
        recipes.append(recipe)
        completeness_counter[sum([i not in ft_vocab for i in recipe])] += 1


print "...which gives complete coverage of",completeness_counter[0],"out of",len(recipes),"recipes. ",completeness_counter[1],"recipes had one missing ingredient, and", \
        len(recipes)-completeness_counter[0]-completeness_counter[1],"had two or more."