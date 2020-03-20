import sys,io,json,urllib,os.path

if __name__ == "__main__":
    recipe_file = sys.argv[1]
    recipe_image_file = sys.argv[2]
    recipe_image_save_path = sys.argv[3]

    with io.open(recipe_file,mode="r",encoding="utf-8") as rf:
        recipe_json = json.loads(rf.read())
    print "Parsed recipes read in."

    with io.open(recipe_image_file,mode="r",encoding="utf-8") as rif:
        recipe_image_json = json.loads(rif.read())
    print "Recipe images read in."

    for recipe in recipe_image_json:
        recipe_image_list = [i["url"] for i in recipe["images"]]
        recipe_json[recipe["id"]]["images"] = recipe_image_list
        for k,img in enumerate(recipe_json[recipe["id"]]["images"]):
            escaped_title = recipe_json[recipe["id"]]["title"].replace("/","")
            urllib.urlretrieve(img,os.path.join(recipe_image_save_path,escaped_title+"_"+str(k)+".jpg"))

    #for recipe_id,recipe in recipe_json.iteritems():
