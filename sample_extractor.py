import sys, json

input_file = sys.argv[1]
output_file = sys.argv[2]
num_to_sample = int(sys.argv[3])

with open(input_file, "r") as f:
    js = json.loads(f.read())
print "** Input file", input_file, "read as json, extracting recipes."

with open(output_file, "w") as fo:
    json.dump(js[:num_to_sample],fo)
print "** Wrote", num_to_sample, "recipes to",output_file+"."
