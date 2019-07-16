import json, os, sys, unicodecsv, tempfile, re, subprocess, io

input_file = sys.argv[1]
stemmed_input = input_file.split(".")[0]
paired_ingredients_file = stemmed_input + "_paired_ingredients.csv"
manual_replacements_file = stemmed_input + "_parsed_ingredient_substitutions.csv"

paired_ingredients = {}
with open(paired_ingredients_file, "rb") as paired_file:
    reader = unicodecsv.reader(paired_file, encoding="utf-8")
    for row in reader:
        if row[1] not in paired_ingredients:
            paired_ingredients[row[1]] = set()
        paired_ingredients[row[1]].add(row[0])

parsed_ings_to_investigate = []
with open(manual_replacements_file, "rb") as replacements_file:
    reader = unicodecsv.reader(replacements_file, encoding="utf-8")
    for row in reader:
        if row[1] == "???":
            parsed_ings_to_investigate.append(row[0])

for ing in parsed_ings_to_investigate:
    print "Investigating",ing
    for input in paired_ingredients[ing]:
        print "  *",input