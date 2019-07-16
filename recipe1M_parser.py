# -*- coding: utf-8 -*-
from ingredient_phrase_tagger.training import utils
import json, os, sys, csv, tempfile, re, subprocess, io

def preparse_replacements(ing):
    units = {
        "lb": u"pound",
        "lbs": u"pounds",
        "c": u"cups",
        "oz": u"ounces",
        "g": u"grams",
        "gm": u"grams",
        "gms": u"grams",
        "tsp": u"teaspoon",
        "tbs": u"tablespoons"
    }

    parsed_ing = re.sub(r'[^\w\s]','',ing).lower()
    if parsed_ing in units.keys():
        return units[parsed_ing]
    else:
        return ing

def tokenise_ingredient(ingredient_phrase):
    clean_phrase = clumpHyphenatedRanges(re.sub('<[^<]+?>,', '', ingredient_phrase))
    tokens = [preparse_replacements(t) for t in utils.tokenize(clean_phrase) if t not in TOKENS_TO_DROP]
    if len(tokens) == 0:
        tokens = ["???"]
        print "Ingredient phrase '"+ingredient_phrase+"' reduced to empty token list"


def clumpHyphenatedRanges(s):
    """
    Replaces the whitespace that's cropping up in things like "1 -2 cups", so it's interpreted as a single token.
    The rest of the string is left alone.

    """
    return re.sub(ur'(\d+)\s*-\s*(\d)', ur'\1-\2', s)

TOKENS_TO_DROP = ["x","pkg","pkg.","minced","finely","thinly","sliced","cl","lrg","lg.", "md.", "med","medium","ml","tablespoons","tablespoon","tbsp","tbsp.","cup.","melted","grated"]

###TO DO: There are four FEWER itmes in the output of this function than there are in the input!  Figure out WTF is going on!
def export_data(lines):
    """ Parse "raw" ingredient lines into CRF-ready output """
    output = []
    for line in lines:
        line_clean = clumpHyphenatedRanges(re.sub('<[^<]+?>,', '', line))
        tokens = [t for t in utils.tokenize(line_clean) if t not in TOKENS_TO_DROP]

        if len(tokens) == 0:
            tokens = ["???"]
            print "?",
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
    print "** NYT-parser input written to txt, calling NYT parser."

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
        raw_ingredients = []
        # loop over each phrase
        failed_ingredients = set()
        print "** Num parsed ingredients found:",len(phrases)
        for phrase in phrases:
            parsed_ing = []
            if len(parsed_ingredients) > 0:
                if len(parsed_ingredients) % 1000 == 0:
                    sys.stdout.write(".")
                    if len(parsed_ingredients) % 100000 == 0:
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
                #print ""
                for word,guess in zip(original,guesses):
                    if not "QTY" in guess and not "UNIT" in guess:
                        parsed_ing.append(word)
                #print 'Parser failed to find an ingredient in: "'+" ".join(parsed_ing)+'".'
                #print phrase
                if len(parsed_ing):
                    failed_ingredients.add(" ".join(parsed_ing))

            #Some ingredients get duplicated by the parser because the name shows up twice,
            #  eg "2 cups fresh spinach or 1 bag frozen spinach" becomes "spinach spinach"
            #  we remove the duplicate.
            if len(parsed_ing) % 2 == 0 and parsed_ing[:len(parsed_ing)/2] == parsed_ing[len(parsed_ing)/2:]:
                parsed_ing = parsed_ing[:len(parsed_ing)/2]

            #Some ingredients contain phrases like "safeway $3 per pound through 09/18", which needs to go away.
            try:
                parsed_ing = parsed_ing[:parsed_ing.index("safeway")]
            except ValueError:
                pass
            parsed_ingredients.append(" ".join(parsed_ing))
            raw_ingredients.append(" ".join(original))
    print ""
    print "** NYT-parsed ingredients read in, writing out failures."

    with io.open(failed_ingredient_output,mode="w", encoding="utf-8") as ff:
        for ing in failed_ingredients:
            ff.write(ing+u"\n")
    print "** Failed parses written to txt, performing substitution."
    return raw_ingredients,parsed_ingredients



if __name__ == "__main__":

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
    manual_replacements_file = stemmed_input + "_parsed_ingredient_substitutions.csv"

    with open(input_file,"r") as f:
        js = json.loads(f.read())
    print "** Input file",input_file,"read as json, extracting recipes."

    unique_ingredient_phrases = set()


    recipes = {}
    for r in js:
        if len(recipes) > 0:
            if len(recipes) % 1000 == 0:
                sys.stdout.write(".")
                if len(recipes) % 100000 == 0:
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
    print "** Num unique ingredients found:",len(unique_ingredient_phrases)
    print "** Recipes and unique ingredients extracted, writing out unique ingredients."

    with open(ingredient_list, "w") as lf:
        writer = csv.writer(lf,delimiter="^",quoting=csv.QUOTE_NONE,escapechar="^")
        for row in unique_ingredient_phrases:
            writer.writerow([row.encode("utf-8")])
    print "** Unique raw ingredients written to CSV, preparing to run NYT parser."

    nyt_parse(ingredient_list, nyt_parser_output)
    print "** NYT-parsed output written to txt, reading results back in."

    preparse_ingredients,parsed_ingredients = nyt_read(nyt_parser_output, failed_ingredient_output)

    # The parser isn't perfect, so we maintain a list of manual post-parse replacements in a file.
    with io.open(manual_replacements_file,mode="r",encoding="utf-8") as rf:
        reader = csv.reader(rf)
        manual_substitutions_dict = {}
        for row in reader:
            manual_substitutions_dict[row[0]] = row[1]
        for k, parsed_ing in enumerate(parsed_ingredients):
            if parsed_ing in manual_substitutions_dict:
                parsed_ingredients[k] = manual_substitutions_dict[parsed_ing]
    print "** Manual substitutions complete, writing out paired raw/parsed ingredients."

    paired_ingredients = zip(preparse_ingredients,parsed_ingredients)
    with open(paired_ingredients_file, "w") as paired_file:
        writer = csv.writer(paired_file)
        for pair in paired_ingredients:
            writer.writerow([pair[0].encode("utf-8"),pair[1].encode("utf-8")])
        print "** Paired raw & parsed ingredients written to CSV, writing out parsed recipes."

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
            # Empty string means we want to drop this ingredient, ??? means I need to investigate and replace it (but drop for now)
            if paired_dict[ing] not in ["???",""]:
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
    print "** Parsed recipes written to json, writing out ingredient frequencies, flat recipes, and vocab."

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
