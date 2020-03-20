# -*- coding: utf-8 -*-
from ingredient_phrase_tagger.training import utils
import json, os, sys, csv, tempfile, re, subprocess, io, unicodecsv, inflection, collections

#Just flagging that {} initialises a set literal, who knew?  I've always used set([])!
AND_OR_INGREDIENT_EXCLUSIONS = {
    "half and half",
    "tomato and herb sauce",
    "sweet and sour mix",
    "garlic and herb seasoning"
}

def separate_and_or_phrases(parsed_ingredients):
    for k,ing in enumerate(parsed_ingredients):
        if type(ing) is not list and ing not in AND_OR_INGREDIENT_EXCLUSIONS:
            if " and " in ing:
                parsed_ingredients[k] = [i.strip() for i in ing.split(" and ")]
            if " or " in ing:
                parsed_ingredients[k] = [i.strip() for i in ing.split(" or ")]
    return parsed_ingredients


def preparse_replacements(ing):
    units = {
        "kg": u"kilograms",
        "lb": u"pound",
        "lbs": u"pounds",
        "c": u"cups",
        "oz": u"ounces",
        "g": u"grams",
        "gm": u"grams",
        "gms": u"grams",
        "tsp": u"teaspoon",
        "tbs": u"tablespoons",
        "&": u"and",
        "qt": u"quart",
        "grnd": u"ground",
        "pwdr": u"powder",
        "chilli": u"pepper",
        "chillies": u"peppers",
        "beetroot": u"beet",
        "yoghurt": u"yogurt",
        "colouring":u"coloring",
        "pitta":u"pita",
        "rapeseed":u"canola"
    }

    parsed_ing = re.sub(r'[^\w\s&]','',ing).lower()
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

TOKENS_TO_DROP = ["free-range","x","pkg","pkg.","minced","finely","thinly","sliced","cl","lrg","lg.", "md.", "med","medium","ml","tablespoons","tablespoon","tbsp","tbsp.","cup.","melted","grated","chopped"]

def tokenise_ingredient(ingredient_phrase):
    tagless_phrase = re.sub('<[^<]+?>,', '', ingredient_phrase)
    bracketfixed_phrase = tagless_phrase.replace("[","(").replace("]",")").replace("{","(").replace("}",")")
    clean_phrase = clumpHyphenatedRanges(bracketfixed_phrase)
    tokens = [preparse_replacements(t) for t in utils.tokenize(clean_phrase) if t.lower() not in TOKENS_TO_DROP]

    # Some ingredients contain emails or phrases like "safeway $3 per pound through 09/18", which need to go away.
    try:
        tokens = tokens[:tokens.index("Safeway")]
    except ValueError:
        pass
    try:
        tokens = tokens[:tokens.index("Target")]
    except ValueError:
        pass
    try:
        tokens = tokens[:tokens.index("Walgreens")]
    except ValueError:
        pass
    try:
        tokens = tokens[:tokens.index("email")]
    except ValueError:
        pass
    try:
        if tokens.index("Whole")+1 == tokens.index("Foods"):
            tokens = tokens[:tokens.index("Whole")]
    except ValueError:
        pass
    try:
        if tokens.index("Rite")+1 == tokens.index("Aid"):
            tokens = tokens[:tokens.index("Rite")]
    except ValueError:
        pass
    try:
        if tokens.index("King")+1 == tokens.index("Sooper's"):
            tokens = tokens[:tokens.index("King")]
    except ValueError:
        pass
    try:
        if tokens.index("Walmart")+1 == tokens.index("Supercenter"):
            tokens = tokens[:tokens.index("Walmart")]
    except ValueError:
        pass
    try:
        if tokens.index("Family")+1 == tokens.index("Dollar"):
            tokens = tokens[:tokens.index("Family")]
    except ValueError:
        pass
    if len(tokens) == 0:
        tokens = ["???"]
        #print "Ingredient phrase '"+ingredient_phrase+"' reduced to empty token list"

    try:
        if tokens.index("Sooper's"):
            print "KEPT:",tokens
    except ValueError:
        pass

    #Some ingredients (from the BBC Recipes database) contain tokens like "55g/2oz" that confuse the NYT parser
    newtokens = []
    for token in tokens:
        if u"/" in token and re.search(ur".*\d.*",token[:token.index(u"/")]) and re.search(ur".*\d.*",token[token.index(u"/"):]):
            pass
        else:
            newtokens.append(token)
    tokens = newtokens

    return tokens

def export_data(lines):
    """ Parse "raw" ingredient lines into CRF-ready output """
    output = []
    for line in lines:
        tokens = line.split()
        for i, token in enumerate(tokens):
            features = utils.getFeatures(token, i+1, tokens)
            output.append(utils.joinLine([token] + features))
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
                    actual_word = line[0]
                    original.append(actual_word)
                    guess = line[-1].split("/")[0]
                    guesses.append(guess)
                    if "NAME" in guess:
                        parsed_ing.append(actual_word)
            if not len(parsed_ing):
                #print ""
                for word,guess in zip(original,guesses):
                    if not "QTY" in guess and not "UNIT" in guess:
                        parsed_ing.append(word)
                #print 'Parser failed to find an ingredient in: "'+" ".join(parsed_ing)+'".'
                #print phrase
                if len(parsed_ing):
                    failed_ingredients.add(" ".join(parsed_ing))

            raw_ingredients.append(" ".join(original))

            #The following is a collection of post-NYT-parse cleanup hackery

            #Some ingredients get duplicated by the parser because the name shows up twice,
            #  eg "2 cups fresh spinach or 1 bag frozen spinach" becomes "spinach spinach"
            #  we remove the duplicate.
            if len(parsed_ing) % 2 == 0 and parsed_ing[:len(parsed_ing)/2] == parsed_ing[len(parsed_ing)/2:]:
                parsed_ing = parsed_ing[:len(parsed_ing)/2]

            #A bunch of ingredients are turning out to look like "can x" and we just want the "x"
            if len(parsed_ing) and parsed_ing[0].lower() in ["can","cans"]:
                parsed_ing = parsed_ing[1:]

            #A bunch of ingredients end with ", undiluted", ditch that
            if len(parsed_ing) > 1 and parsed_ing[-2:] == [",","undiluted"]:
                parsed_ing = parsed_ing[:-2]

            #A bunch of ingredients are turning up ending in :, and those are all sub-headings within the ingredients list, so we can skip them.
            if len(parsed_ing) and parsed_ing[-1][-1] == u":":
                parsed_ing = [""]

            new_ingredient = " ".join(parsed_ing)
            #new_ingredient = inflection.singularize(new_ingredient)
            parsed_ingredients.append(new_ingredient)
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
    ingredient_list = stemmed_input+"_ingredients.txt"
    nyt_parser_output = stemmed_input + "_ingredients_nytparser_output.txt"
    paired_ingredients_file = stemmed_input + "_paired_ingredients.csv"
    recipes_out = stemmed_input + "_parsed_recipes.json"
    failed_ingredient_output = stemmed_input + "_failed_ingredients.txt"
    flat_recipes = stemmed_input + "_flat_recipes.txt"
    ingredient_freqs_out = stemmed_input + "_parsed_ingredient_freqs.csv"
    vocab_out = stemmed_input + "_vocab.txt"
    manual_replacements_file = stemmed_input + "_parsed_ingredient_substitutions.csv"

    singularise = False
    if len(sys.argv) > 2:
        singularise = bool(sys.argv[2])

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
        recipe = {u"title":r["title"],u"raw_ingredients":[],u"instructions":[],u"tokenised_ingredients":[]}
        ings = r["ingredients"]
        for i in ings:
            ing = i["text"]
            recipe[u"raw_ingredients"].append(ing)
            tokenised_ing = tokenise_ingredient(ing)
            recipe[u"tokenised_ingredients"].append(tokenised_ing)
            unique_ingredient_phrases.add(" ".join(tokenised_ing))
        steps = r["instructions"]
        for s in steps:
            recipe[u"instructions"].append(s["text"])
        recipes[r["id"]] = recipe
    print ""
    print "** Num unique ingredients found:",len(unique_ingredient_phrases)
    print "** Recipes and unique ingredients extracted, writing out unique ingredients."

    with io.open(ingredient_list, mode="w",encoding="utf-8") as lf:
        for phrase in unique_ingredient_phrases:
            lf.write(phrase+u"\n")
    print "** Unique preparsed ingredients written to txt, preparing to run NYT parser."

    nyt_parse(ingredient_list, nyt_parser_output)
    print "** NYT-parsed output written to txt, reading results back in."

    preparse_ingredients,parsed_ingredients = nyt_read(nyt_parser_output, failed_ingredient_output)

    # The parser isn't perfect, so we maintain a list of manual post-parse replacements in a file.
    try:
        with open(manual_replacements_file,"r") as rf:
            reader = unicodecsv.reader(rf)
            manual_substitutions_dict = {}
            for row in reader:
                if len(row) == 2:
                    manual_substitutions_dict[row[0].lower()] = row[1].lower()
                else:
                    manual_substitutions_dict[row[0].lower()] = [r.lower() for r in row[1:]]
            for k, parsed_ing in enumerate(parsed_ingredients):
                if parsed_ing.lower() in manual_substitutions_dict:
                    parsed_ingredients[k] = manual_substitutions_dict[parsed_ing.lower()]
                elif "Chinese black" in parsed_ing:
                    a = preparse_ingredients[k]
                    pass
        print "** Manual substitutions complete, writing out paired raw/parsed ingredients."
    except IOError:
        print "** No manual substitution file found, skipping to writing out paired raw/parsed ingredients."


    parsed_ingredients = separate_and_or_phrases(parsed_ingredients)
    print '** Separated ingredients with "and" or "or" in them.'

    if singularise:
        for k,ing in enumerate(parsed_ingredients):
            if type(ing) is list:
                parsed_ingredients[k] = [inflection.singularize(i) for i in ing]
            else:
                parsed_ingredients[k] = inflection.singularize(ing)
        print '** Singularised all ingredients.'

    paired_ingredients = zip(preparse_ingredients,parsed_ingredients)
    with open(paired_ingredients_file, "w") as paired_file:
        writer = csv.writer(paired_file)
        for pair in paired_ingredients:
            if type(pair[1]) is list:
                writer.writerow([pair[0].encode("utf-8"), [p.encode("utf-8") for p in pair[1]]])
            else:
                writer.writerow([pair[0].encode("utf-8"), pair[1].encode("utf-8")])
        print "** Paired raw & parsed ingredients written to CSV, writing out parsed recipes."

    ingredient_freqs = {}
    paired_dict = dict(paired_ingredients)
    done = 0
    completeness_counter = collections.Counter()
    for id,recipe in recipes.iteritems():
        recipe[u"ingredients"] = []
        recipe[u"missing"] = 0
        if done > 0:
            if done % 1000 == 0:
                sys.stdout.write(".")
                if done % 100000 == 0:
                    print ""
        for ing_tokens in recipe[u"tokenised_ingredients"]:
            ing = " ".join(ing_tokens)
            # Empty string means we want to drop this ingredient, ??? means I need to investigate and replace it (but drop for now)
            if ing in paired_dict:
                if type(paired_dict[ing]) is not list:
                    parsed_ings = [paired_dict[ing].lower()]
                else:
                    parsed_ings = [i.lower() for i in paired_dict[ing]]
                for parsed_ing in parsed_ings:
                    if parsed_ing not in ["???",""]:
                        if parsed_ing not in ingredient_freqs:
                            ingredient_freqs[parsed_ing] = 1
                        else:
                            ingredient_freqs[parsed_ing] += 1
                        recipe[u"ingredients"].append(parsed_ing)
            else:
                recipe[u"missing"] += 1
                print ing
        done += 1
        completeness_counter[recipe[u"missing"]] +=1
    print ""
    with io.open(recipes_out,mode="w",encoding="utf-8") as rf:
        j = json.dumps(recipes,rf, ensure_ascii=False)
        rf.write(j)
    print "** Parsed recipes written to json, writing out ingredient frequencies, flat recipes, and vocab.",completeness_counter[0],\
        "recipes had zero ingredients missing,",completeness_counter[1],"recipes had one, and",len(recipes)-completeness_counter[0]-completeness_counter[1],\
        "recipes had two or more."

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
