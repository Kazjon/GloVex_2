import sys, io, csv

from recipe1M_parser import nyt_parse,nyt_read

if __name__ == "__main__":

    failed_ingredient_input_file = sys.argv[1]
    stemmed_input = failed_ingredient_input_file.split(".")[0]
    reparsed_output = stemmed_input+"_reparsed.txt"
    refailed_output = stemmed_input+"_refailed.txt"
    paired_ingredients_file = stemmed_input + "_paired_failed_ingredients.csv"

    nyt_parse(failed_ingredient_input_file,reparsed_output)
    parsed_ingredients = nyt_read(reparsed_output,refailed_output)

    ingredient_phrases = []
    with io.open(failed_ingredient_input_file, mode="r", encoding="utf-8") as ff:
        for row in ff:
            ingredient_phrases.append(row)

    paired_ingredients = zip(ingredient_phrases,parsed_ingredients)
    with open(paired_ingredients_file, "w") as paired_file:
        writer = csv.writer(paired_file)
        for pair in paired_ingredients:
            writer.writerow([pair[0].encode("utf-8"),pair[1].encode("utf-8")])
        print "** Paired raw & parsed ingredients written to CSV."