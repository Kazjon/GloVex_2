import json, os, sys, unicodecsv, tempfile, re, subprocess, io

weird_ingredients = [
                        "of",
                        "extract",
                        "for garnish",
                        "black",
                        "cooking",
                        "mayer",
                        "on-the-go packet",
                        "bake",
                        "pepperidge",
                        "flavoring",
                        "pouch",
                        "env.",
                        "pkt",
                        "green",
                        "canned",
                        "added",
                        "breakstone's free",
                    ]

input_file = sys.argv[1]
stemmed_input = input_file.split(".")[0]
paired_ingredients_file = stemmed_input + "_paired_ingredients.csv"
manual_replacements_file = stemmed_input + "_parsed_ingredient_substitutions.csv"

paired_ingredients = {}
with open(paired_ingredients_file, "rb") as paired_file:
    reader = unicodecsv.reader(paired_file, encoding="utf-8")
    for row in reader:
        if row[1].lower() not in paired_ingredients:
            paired_ingredients[row[1].lower()] = set()
        paired_ingredients[row[1].lower()].add(row[0])

parsed_ings_to_investigate = []
#with open(manual_replacements_file, "rb") as replacements_file:
#    reader = unicodecsv.reader(replacements_file, encoding="utf-8")
#    for row in reader:
#        if row[1] == "???":
#            parsed_ings_to_investigate.append(row[0])
parsed_ings_to_investigate = weird_ingredients

for ing in parsed_ings_to_investigate:
    print "Investigating",ing
    if ing in paired_ingredients:
        for input in paired_ingredients[ing]:
            print "  *",input
    else:
        print "  *",ing,"not found in",paired_ingredients_file