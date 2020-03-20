import csv, sys, json, io, re, os
import numpy as np

subcluster_input_file = sys.argv[1]
vocab_input_file = sys.argv[2]
cluster_labels_file = sys.argv[3]
recipe_input_file = sys.argv[4]
original_recipe_input_file = sys.argv[5]

recipe_output_file = sys.argv[6]
ingredients_output_file = sys.argv[7]
subcluster_output_file = sys.argv[8]
cluster_output_file = sys.argv[9]

ingredients = {}

with open(vocab_input_file, "r") as f:
    reader = csv.reader(f)
    for k,row in enumerate(reader):
        ingredients[row[0]] = {"id":k,"frequency":int(row[1])}

subcluster_labels = {}

with open(cluster_labels_file, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        subcluster_labels[int(row[0])] = row[1]

clusters = sorted(set(subcluster_labels.values()))

subclusters = {}

ingredients_output_json = {}

with open(subcluster_input_file, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        ing = row[0]
        subcluster = int(row[1])
        if subcluster == -2:
            continue
        elif subcluster == -3:
            subcluster = None
            cluster = None
        else:
            if subcluster not in subclusters:
                subclusters[subcluster] = [ing]
            else:
                subclusters[subcluster].append(ing)
            cluster = clusters.index(subcluster_labels[subcluster])

        ing_json = {"name":ing,"subcluster":subcluster,"cluster":cluster,"frequency":ingredients[ing]["frequency"]}
        ingredients_output_json[ingredients[ing]["id"]] = ing_json
        #print ing_json

with open(ingredients_output_file, "w") as f:
   json.dump(ingredients_output_json,f)

subclusters_output_json = {}
cluster_members = {}

for id,cluster in subcluster_labels.iteritems():
    if id in subclusters:
        if cluster in cluster_members:
            cluster_members[cluster].append(id)
        else:
            cluster_members[cluster] = [id]
        ings = subclusters[id]
        freqs = [ingredients[ing]["frequency"] for ing in ings]
        highest_freq_ing = np.argmax(freqs)
        name = ings[highest_freq_ing]+"_subcluster"
        cluster_id = clusters.index(cluster)
        subcluster_json =  {"name":name,"cluster":cluster_id,"ingredients":ings,"frequency":sum(freqs)}
        subclusters_output_json[id] = subcluster_json
        #print subcluster_json

with open(subcluster_output_file, "w") as f:
   json.dump(subclusters_output_json,f)

cluster_output_json = {}

for id,cluster in enumerate(clusters):
    subs_of_this_cluster = cluster_members[cluster]
    ings = [i for scid in subs_of_this_cluster for i in subclusters[scid]]
    freqs = [ingredients[ing]["frequency"] for ing in ings]
    cluster_json = {"name":cluster,"subclusters":subs_of_this_cluster,"ingredients":[ingredients[i]["id"] for i in ings],"frequency":sum(freqs)}
    cluster_output_json[id] = cluster_json
    #print cluster_json


with open(cluster_output_file, "w") as f:
   json.dump(cluster_output_json,f)

with io.open(recipe_input_file,mode="r",encoding="utf-8") as rf:
    recipe_input_json = json.loads(rf.read())


with io.open(original_recipe_input_file,mode="r",encoding="utf-8") as rf:
    original_recipe_input_json = json.loads(rf.read())

recipe_output_json = {}

for bbc_recipe_id,recipe in recipe_input_json.iteritems():
    recipe_id = int(bbc_recipe_id.split("_")[-1])
    original_recipe = original_recipe_input_json[bbc_recipe_id]
    ingredient_ids = [ingredients[re.sub(ur'\s+', u'_',i)]["id"] for i in recipe["ingredients"]]
    recipe_json = {"title":recipe["title"],
                   "url":original_recipe["url"],
                   "ingredient_phrases":recipe["raw_ingredients"],
                   "ingredient_ids":ingredient_ids,
                   "steps":recipe["instructions"],
                   "image":os.path.split(original_recipe["image"])[1],
                   "servings":original_recipe["serves"],
                   "vegetarian":original_recipe["isVegetarian"],
                   "prepTime":original_recipe["time"]["preparationMins"],
                   "cookTime":original_recipe["time"]["cookingMins"]
                   }
    recipe_output_json[recipe_id] = recipe_json

with open(recipe_output_file, "w") as f:
   json.dump(recipe_output_json,f)
