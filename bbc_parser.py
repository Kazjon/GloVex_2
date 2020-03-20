# -*- coding: utf-8 -*-
import sys, json, io, time, urllib, random, os, unicodecsv


unicode_replacements = {
    u"¼":u".25",
    u"½":u".5",
    u"¾":u".75",
}

def bbc_preparse(ing):
    ing = unicode(ing)
    ing = ing.strip()
    #ing = ing.replace("/"," / ")
    for k,v in unicode_replacements.iteritems():
        ing = ing.replace(k,v)
        
    return ing


download_images = False
output_titles = True
discard_excluded_recipes = True

if __name__ == "__main__":

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    recipe_image_path = sys.argv[3]
    titles_output_path = sys.argv[4]
    exclusions_path = sys.argv[5]

    with io.open(input_file, mode='r', encoding="utf-8") as f:
        js = json.loads(f.read())
    print "** Input file",input_file,"read as json, extracting recipes."

    recipes_to_discard = []
    if discard_excluded_recipes:
        try:
            with io.open(exclusions_path,mode="rU",encoding="utf-8-sig",errors="ignore") as rf:
                for id,line in enumerate(rf):
                    line = line.rstrip()
                    line = line.replace("_","")
                    if line[0] == '"':
                        row = [line[1:line.rfind('"')],line[line.rfind(","):]]
                    else:
                        row = line.split(",")
                    print row
                    if len(row[1]):
                        recipes_to_discard.append(unicode.encode(row[0],errors="ignore"))
            print "** Manual recipe exclusions complete, writing out remaining parsed recipes."
        except IOError:
            print "** No manual recipe exclusions file found, skipping to writing out parsed recipes."
    recipes_to_discard = set(recipes_to_discard)

    output_recipes = []
    for id,recipe in js.iteritems():
        if "image" in recipe.keys():
            if not unicode.encode(recipe["title"],errors="ignore") in recipes_to_discard:
                #print recipe
                out_recipe = {u"id":unicode(id),u"url":recipe["url"],u"title":recipe["title"]}
                out_recipe[u"instructions"] = [{u"text":unicode(instruction.strip())} for instruction in recipe["method"]]
                out_recipe[u"ingredients"] = [{u"text":bbc_preparse(ing_string)} for ing_string in recipe["ingredients"]]
                #if "time" in recipe.keys():
                out_recipe[u"prep_time"] = recipe["time"]["preparationMins"]
                out_recipe[u"cook_time"] = recipe["time"]["cookingMins"]
                out_recipe[u"vegetarian"] = recipe["isVegetarian"]
                out_recipe[u"servings"] = recipe["serves"]
                out_recipe[u"image"] = recipe["image"].replace("16x9_448","16x9_832")
                #print out_recipe
                #print "*******"
                #print
                output_recipes.append(out_recipe)


    with io.open(output_file, mode='w', encoding="utf-8") as f:
        f.write(json.dumps(output_recipes, ensure_ascii=False))
    print "** Wrote "+str(len(output_recipes))+" recipes to "+output_file+"."

    if download_images:
        for i,recipe in enumerate(output_recipes):
            urllib.urlretrieve(recipe[u"image"], os.path.join(recipe_image_path, recipe[u"title"] + ".jpg"))
            time.sleep(0.5+random.random())
            if i % 100 == 0:
                print ".",

    if output_titles:
        titles = ['"'+r["title"]+'",\n' for r in output_recipes]
        with io.open(titles_output_path, mode='w', encoding="utf-8") as f:
            f.writelines(titles)
        print "** Wrote " + str(len(output_recipes)) + " recipe titles to " + titles_output_path + "."

    u'''

    #Input sample:
    {"www_bbc_co_uk_food_recipes_15_minute_pasta_33407": {
        "ingredients": [" 350g/12oz penne pasta ", " 2 x 80g/3oz packs Parma ham, snipped into small pieces", " 250g/9oz small brown chestnut mushrooms, halved or quartered",
                        " 200g/7oz full-fat crème fraîche", " 100g/3½oz Parmesan, grated ", " 2 tbsp chopped parsley ", " salt and pepper, to taste", " green salad",
                        " crunchy bread"], "method": ["Cook the pasta in a pan of boiling salted water according to the packet instructions. Drain and set aside",
                                                      "Heat a frying pan until hot. Add the pieces of Parma ham and fry until crisp, remove half of the ham onto a plate and set aside. Add the mushrooms to the pan and fry for two minutes. Add the crème fraîche and bring up to the boil. Add the pasta, Parmesan and parsley and toss together over the heat. Season well with salt and pepper.",
                                                      "Serve with a green salad and crunchy bread."], "url": "www_bbc_co_uk_food_recipes_15_minute_pasta_33407",
        "title": "15 minute pasta", "time": {"preparation": "less than 30 mins", "preparationMins": 30, "cooking": "10 to 30 mins", "cookingMins": 30, "totalMins": 60},
        "serves": "Serves 6", "image": "http://ichef.bbci.co.uk/food/ic/food_16x9_448/recipes/15_minute_pasta_33407_16x9.jpg", "isVegetarian": false, "recommendations": 0}
    }

    #Output sample:
    {"title": "Worlds Best Mac and Cheese", "url": "http://www.epicurious.com/recipes/food/views/-world-s-best-mac-and-cheese-387747", "partition": "train",
      "ingredients": [{"text": "6 ounces penne"}, {"text": "2 cups Beechers Flagship Cheese Sauce (recipe follows)"}, {"text": "1 ounce Cheddar, grated (1/4 cup)"},
                      {"text": "1 ounce Gruyere cheese, grated (1/4 cup)"}, {"text": "1/4 to 1/2 teaspoon chipotle chili powder (see Note)"},
                      {"text": "1/4 cup (1/2 stick) unsalted butter"}, {"text": "1/3 cup all-purpose flour"}, {"text": "3 cups milk"},
                      {"text": "14 ounces semihard cheese (page 23), grated (about 3 1/2 cups)"}, {"text": "2 ounces semisoft cheese (page 23), grated (1/2 cup)"},
                      {"text": "1/2 teaspoon kosher salt"}, {"text": "1/4 to 1/2 teaspoon chipotle chili powder"}, {"text": "1/8 teaspoon garlic powder"},
                      {"text": "(makes about 4 cups)"}], "id": "000018c8a5",
      "instructions": [{"text": "Preheat the oven to 350 F. Butter or oil an 8-inch baking dish."}, {"text": "Cook the penne 2 minutes less than package directions."},
                       {"text": "(It will finish cooking in the oven.)"}, {"text": "Rinse the pasta in cold water and set aside."},
                       {"text": "Combine the cooked pasta and the sauce in a medium bowl and mix carefully but thoroughly."},
                       {"text": "Scrape the pasta into the prepared baking dish."}, {"text": "Sprinkle the top with the cheeses and then the chili powder."},
                       {"text": "Bake, uncovered, for 20 minutes."}, {"text": "Let the mac and cheese sit for 5 minutes before serving."},
                       {"text": "Melt the butter in a heavy-bottomed saucepan over medium heat and whisk in the flour."}, {"text": "Continue whisking and cooking for 2 minutes."},
                       {"text": "Slowly add the milk, whisking constantly."}, {"text": "Cook until the sauce thickens, about 10 minutes, stirring frequently."},
                       {"text": "Remove from the heat."}, {"text": "Add the cheeses, salt, chili powder, and garlic powder."},
                       {"text": "Stir until the cheese is melted and all ingredients are incorporated, about 3 minutes."},
                       {"text": "Use immediately, or refrigerate for up to 3 days."}, {"text": "This sauce reheats nicely on the stove in a saucepan over low heat."},
                       {"text": "Stir frequently so the sauce doesnt scorch."}, {
                           "text": "This recipe can be assembled before baking and frozen for up to 3 monthsjust be sure to use a freezer-to-oven pan and increase the baking time to 50 minutes."},
                       {"text": "One-half teaspoon of chipotle chili powder makes a spicy mac, so make sure your family and friends can handle it!"},
                       {"text": "The proportion of pasta to cheese sauce is crucial to the success of the dish."},
                       {"text": "It will look like a lot of sauce for the pasta, but some of the liquid will be absorbed."}]}

    #'''