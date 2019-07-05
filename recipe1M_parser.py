# -*- coding: utf-8 -*-
from ingredient_phrase_tagger.training import utils
from pattern.en import singularize
import json, os, sys, csv, tempfile, re, subprocess, io
import cPickle as pickle

def preparse_replacements(ing):
    units = {
        "lb": u"pound",
        "lbs": u"pounds",
        "c": u"cups",
        "oz": u"ounces",
        "g": u"grams"

    }

    parsed_ing = re.sub(r'[^\w\s]','',ing).lower()
    if parsed_ing in units.keys():
        return units[parsed_ing]
    else:
        return ing

def clumpHyphenatedRanges(s):
    """
    Replaces the whitespace that's cropping up in things like "1 -2 cups", so it's interpreted as a single token.
    The rest of the string is left alone.

    """
    return re.sub(ur'(\d+)\s*-\s*(\d)', ur'\1-\2', s)

TOKENS_TO_DROP = ["x"]

def export_data(lines):
    """ Parse "raw" ingredient lines into CRF-ready output """
    output = []
    for line in lines:
        line_clean = clumpHyphenatedRanges(re.sub('<[^<]+?>,', '', line))
        tokens = [t for t in utils.tokenize(line_clean) if t not in TOKENS_TO_DROP]

        for i, token in enumerate(tokens):
            modified_token = preparse_replacements(token)
            features = utils.getFeatures(modified_token, i+1, tokens)
            output.append(utils.joinLine([modified_token] + features))
        output.append('')
    return u'\n'.join(output)


def nyt_parse(ingredient_list, nyt_parser_output):
    _, tmpFile = tempfile.mkstemp()

    with io.open(ingredient_list, mode='r',encoding="utf-8") as infile, io.open(tmpFile, mode='w',encoding="utf-8") as outfile:
        outfile.write(export_data(infile.readlines()))

    tmpFilePath = "ingredient-phrase-tagger/tmp/model_file"
    modelFilename = os.path.join(os.getcwd(), tmpFilePath)
    with open(nyt_parser_output, "w") as nyt_out:
        subprocess.call("crf_test -v 1 -m %s %s" % (modelFilename, tmpFile), shell=True, stdout=nyt_out)
    os.system("rm %s" % tmpFile)

def nyt_read(nyt_parser_output,failed_ingredient_output):
    #Read in the CRF file and turn it into a list of names
    with io.open(nyt_parser_output, mode="r", encoding="utf-8") as crf_in:
        phrases = crf_in.read().split('\n\n')

        parsed_ingredients = []
        # loop over each phrase
        done = 0
        failed_ingredients = set()
        for phrase in phrases:
            parsed_ing = []
            if done % 10000 == 0:
                print ".",
                if done > 0 and done % 1000000 == 0:
                    print ""
            original = []
            guesses = []
            for word in phrase.split('\n'):
                line = word.strip().split('\t')

                if len(line) > 1:
                    word = line[0].lower()
                    original.append(word)
                    guess = line[-1].split("/")[0]
                    guesses.append(guess)
                    if "NAME" in guess:
                        parsed_ing.append(word)
            if not len(parsed_ing):
                print ""
                for word,guess in zip(original,guesses):
                    if not "QTY" in guess and not "UNIT" in guess:
                        parsed_ing.append(word)
                print 'Parser failed to find an ingredient in: "'+" ".join(parsed_ing)+'".'
                failed_ingredients.add(" ".join(parsed_ing))
            parsed_ingredients.append(" ".join(parsed_ing))
    print ""

    with io.open(failed_ingredient_output,mode="w", encoding="utf-8") as ff:
        for ing in failed_ingredients:
            ff.write(ing+u"\n")
    print "** Failed parses written to txt."
    return parsed_ingredients



if __name__ == "__main__":

    WRITE_RAWS = False
    WRITE_NYT = False
    WRITE_RECIPES = True

    input_file = sys.argv[1]
    stemmed_input = input_file.split(".")[0]
    pickled_input = stemmed_input+".pkl"
    ingredient_list = stemmed_input+"_ingredients.csv"
    nyt_parser_output = stemmed_input + "_ingredients_nytparser_output.txt"
    paired_ingredients_file = stemmed_input + "_paired_ingredients.csv"
    recipes_out = stemmed_input + "_parsed_recipes.json"
    failed_ingredient_output = stemmed_input + "_failed_ingredients.txt"
    flat_recipes = stemmed_input + "_flat_recipes.txt"
    ingredient_freqs_out = stemmed_input + "_parsed_ingredient_freqs.csv"
    vocab_out = stemmed_input + "_vocab.txt"

    with open(input_file,"r") as f:
        js = json.loads(f.read())
    print "** Input file",input_file,"read as json."

    unique_ingredient_phrases = set()


    recipes = {}
    for r in js:
        if len(recipes) % 1000 == 0:
            print ".",
            if len(recipes) > 0 and len(recipes) % 100000 == 0:
                print ""
        recipe = {u"title":r["title"],u"raw_ingredients":[],u"instructions":[]}
        ings = r["ingredients"]
        for i in ings:
            ing = i["text"]
            recipe[u"raw_ingredients"].append(ing)
            unique_ingredient_phrases.add(ing)
        steps = r["instructions"]
        for s in steps:
            recipe[u"instructions"].append(s["text"])
        recipes[r["id"]] = recipe
    print ""
    print "** Recipes and unique ingredients extracted."

    if WRITE_RAWS:
        with open(ingredient_list, "w") as lf:
            writer = csv.writer(lf,delimiter="^",quoting=csv.QUOTE_NONE,escapechar="^")
            for row in unique_ingredient_phrases:
                writer.writerow([row.encode("utf-8")])
        print "** Unique raw ingredients written to CSV."

    if WRITE_NYT:
        nyt_parse(ingredient_list, nyt_parser_output)
        print "** NYT-parsed output written to txt."

    parsed_ingredients = nyt_read(nyt_parser_output, failed_ingredient_output)

    #assert len(unique_ingredient_phrases) == len(parsed_ingredients)
    paired_ingredients = zip(unique_ingredient_phrases,parsed_ingredients)
    with open(paired_ingredients_file, "w") as paired_file:
        writer = csv.writer(paired_file)
        for pair in paired_ingredients:
            writer.writerow([pair[0].encode("utf-8"),pair[1].encode("utf-8")])
        print "** Paired raw & parsed ingredients written to CSV."

    if WRITE_RECIPES:
        ingredient_freqs = {}
        paired_dict = dict(paired_ingredients)
        done = 0
        for id,recipe in recipes.iteritems():
            recipe[u"ingredients"] = []
            if done % 1000 == 0:
                print ".",
                if done > 0 and done % 100000 == 0:
                    print ""
            for ing in recipe[u"raw_ingredients"]:
                if paired_dict[ing] not in ingredient_freqs:
                    ingredient_freqs[paired_dict[ing]] = 1
                else:
                    ingredient_freqs[paired_dict[ing]] += 1
                recipe[u"ingredients"].append(paired_dict[ing])
            done += 1
        print ""
        with io.open(recipes_out,mode="w",encoding="utf-8") as rf:
            j = json.dumps(recipes,rf, ensure_ascii=False)
            rf.write(j)
        print "** Parsed recipes written to json."
        with io.open(ingredient_freqs_out,mode="w",encoding="utf-8") as fqf:
            for k,v in sorted(ingredient_freqs.items(),key=lambda x: x[1],reverse=True):
                fqf.write(u'"'+k+u'",'+unicode(v)+u"\n")
        print "** Ingredient frequencies written to csv."
        with io.open(flat_recipes,mode="w",encoding="utf-8") as flf:
            for id,recipe in recipes.iteritems():
                flf.write(u" ".join([re.sub(ur'\s+', u'_', i) for i in recipe[u"ingredients"]])+u"\n")
        print "** Flat recipes written to txt."
        with io.open(vocab_out,mode="w",encoding="utf-8") as fqf:
            #fqf.write(u"\n".join(["[PAD]","[CLS]","[SEP]","[UNK]","[MASK]"])+u"\n")
            for k,v in sorted(ingredient_freqs.items(),key=lambda x: x[1],reverse=True):
                fqf.write(re.sub(ur'\s+', u'_', k)+u"\n")
        print "** Vocab file written to txt."
