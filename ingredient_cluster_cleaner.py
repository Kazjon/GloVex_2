import csv, sys, os, copy

original_cluster_file = sys.argv[1]
modified_cluster_file = sys.argv[2]
clusters_checked_file = sys.argv[3]
cluster_labels_file = sys.argv[4]
clusters_reviewed_file = sys.argv[5]




#####Loading in stuff
def load_clusters():
    ingredients = set([])
    original_clusters = {}
    with open(original_cluster_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            ing = row[0]
            ingredients.add(ing)
            cluster = int(row[1])
            if cluster not in original_clusters:
                original_clusters[cluster] = [ing]
            else:
                original_clusters[cluster].append(ing)

    new_clusters = {}
    if os.path.exists(modified_cluster_file):
        with open(modified_cluster_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                ing = row[0]
                cluster = int(row[1])
                if cluster not in new_clusters:
                    new_clusters[cluster] = [ing]
                else:
                    new_clusters[cluster].append(ing)
    else:
        new_clusters = copy.deepcopy(original_clusters)

    if not -2 in new_clusters.keys():
        new_clusters[-2] = []

    clusters_checked = []
    if os.path.exists(clusters_checked_file):
        with open(clusters_checked_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                clusters_checked.append(int(row[0]))

    cluster_labels = {}

    if os.path.exists(cluster_labels_file):
        with open(cluster_labels_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                cluster_labels[int(row[0])] = row[1]


    clusters_reviewed = []
    if os.path.exists(clusters_reviewed_file):
        with open(clusters_reviewed_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                clusters_reviewed.append(row[0])

    return ingredients, original_clusters, new_clusters, clusters_checked, cluster_labels, clusters_reviewed


#####Writing out stuff
def save_clusters(original_clusters, new_clusters, clusters_checked, cluster_labels, verbose=True):
    new_clusters_by_ing = []
    for cluster_id, cluster in new_clusters.iteritems():
        for ing in cluster:
            new_clusters_by_ing.append((ing, cluster_id))

    with open(modified_cluster_file, "w") as f:
        writer = csv.writer(f)
        for ing, clustering in new_clusters_by_ing:
            writer.writerow([ing, clustering])

    with open(cluster_labels_file, "w") as f:
        writer = csv.writer(f)
        for cluster, label in cluster_labels.iteritems():
            writer.writerow([cluster, label])

    with open(clusters_checked_file, "w") as f:
        writer = csv.writer(f)
        for cluster in clusters_checked:
            writer.writerow([cluster])

    if verbose:
        print "Progress saved. Completed",len(clusters_checked),"of",len(original_clusters.keys()),"clusters. ("+str(100*round(len(clusters_checked)/float(len(original_clusters.keys())),4))+"%)."

def save_clusters_post_review(clusters_reviewed, verbose=True):
    with open(clusters_reviewed_file, "w") as f:
        writer = csv.writer(f)
        for cluster in clusters_reviewed:
            writer.writerow([cluster])

    if verbose:
        print "Review progress saved. Reviewed", len(clusters_reviewed), "of", len(inverted_cluster_labels.keys()), "clusters. (" + str(
            100 * round(len(clusters_reviewed) / float(len(inverted_cluster_labels.keys())), 4)) + "%)."

def edit_cluster(id,cluster, new_clusters):
    delete_response = ""
    while delete_response is "":
        delete_response = raw_input("Which, if any, ingredients are not food? (list IDs above, separated by spaces, leave blank for none)")
        try:
            delete_response = [int(r) for r in delete_response.split()]
        except:
            delete_response = ""
            print "Sorry, didn't quite catch that."
    for ing_id in sorted(delete_response, reverse=True):
        ing = cluster[ing_id]
        if id in new_clusters and ing in new_clusters[id]:
            new_clusters[id].remove(ing)
        if not ing in new_clusters[-2]:
            new_clusters[-2].append(ing)
    mismatch_response = ""
    while mismatch_response is "":
        mismatch_response = raw_input("Which, if any, ingredients do NOT fit the cluster's theme? (list IDs above, separated by spaces, leave blank for none)")
        try:
            mismatch_response = [int(r) for r in mismatch_response.split()]
        except:
            mismatch_response = ""
            print "Sorry, didn't quite catch that."
    for ing_id in sorted(mismatch_response, reverse=True):
        ing = cluster[ing_id]
        if id in new_clusters and ing in new_clusters[id]:
            new_clusters[id].remove(ing)
        if not ing in new_clusters[-1]:
            new_clusters[-1].append(ing)

#####Actually doing stuff

ingredients, original_clusters, new_clusters, clusters_checked, cluster_labels, clusters_reviewed = load_clusters()

for id,cluster in original_clusters.iteritems():
    if not id == -1 and id not in clusters_checked:
        print ", ".join([str(index)+":"+ing for index,ing in zip(range(len(cluster)),cluster)])
        response = ""
        while response == "":
            response = raw_input("Is this cluster thematically coherent? (y/n or q to exit)")
            if response not in ["y","n","q","yes","no","quit"]:
                response = ""
                print "Sorry, didn't quite catch that."
        if response.lower() in ["y","yes"]:
            edit_cluster(id,cluster,new_clusters)
            available_labels = set(cluster_labels.values())
            if len(available_labels):
                print "Current clusters available for merging are:",", ".join(sorted(available_labels))
            else:
                print "There are currently no named and verified clusters."
            response = raw_input("What name would you give this cluster? (will merge if a name is picked from above, otherwise will create a new cluster)")
            if response in available_labels:
                print "Adding the current ingredients to existing cluster",response
            else:
                print "Creating a new cluster, `"+response+"'."
            cluster_labels[id] = response
            clusters_checked.append(id)
        elif response == "q":
            print "Exiting."
            sys.exit()

        else: #Destroy this cluster
            for ing in new_clusters[id]:
                new_clusters[-1].append(ing)
            del new_clusters[id]
            clusters_checked.append(id)
            print "Cluster deleted."
        save_clusters(original_clusters, new_clusters, clusters_checked, cluster_labels)
        print
        print


print "No clusters remaining to search. Named clusters:",", ".join(sorted(set(cluster_labels.values())))

if -1 in new_clusters:
    remaining_unclustered = copy.copy(new_clusters[-1])
else:
    remaining_unclustered = []
inverted_cluster_labels = {}
for id,label in cluster_labels.iteritems():
    if not label in inverted_cluster_labels:
        inverted_cluster_labels[label] = [id]
    else:
        inverted_cluster_labels[label].append(id)

if not -3 in new_clusters:
    new_clusters[-3] = [] #These are the ingredients we have confirmed should not be clustered.
for ing in remaining_unclustered:
    new_clusters[-1].remove(ing)
    response = None
    while response == None:
        response = raw_input(str(len(new_clusters[-1]))+" unclustered ingredients remaining. To which cluster should '"+ing+"' be assigned? (blank to leave unclustered, "
                                                                                                                            "'d' to delete, 'q' to quit):")
        if len(response):
            if response == 'd':
                new_clusters[-2].append(ing)
            elif response == 'q':
                print "Exiting."
                sys.exit()
            elif response in inverted_cluster_labels:
                new_clusters[inverted_cluster_labels[response][0]].append(ing)
            else:
                new_cluster_response = ""
                while new_cluster_response == "":
                    new_cluster_response = raw_input("Sorry, that isn't a cluster. Do you want to create a new cluster by that name? (y/n):")
                    if new_cluster_response == "y":
                        new_id = max(cluster_labels.keys())+1
                        new_clusters[new_id] = [ing]
                        cluster_labels[new_id] = response
                        inverted_cluster_labels[response] = new_id
                    elif new_cluster_response == "n":
                        response = None
                    else:
                        print "Sorry, didn't catch that. ",
                        new_cluster_response = ""
        else:
            new_clusters[-3].append(ing)
    save_clusters(original_clusters, new_clusters, clusters_checked, cluster_labels, verbose=False)

if -1 not in new_clusters:
    new_clusters[-1] = []

print "All ingredients have been assigned to clusters or verified as singletons. Reviewing clusters."
for cluster_label,subcluster_ids in sorted(inverted_cluster_labels.iteritems(),key=lambda x:x[0]):
    if cluster_label not in clusters_reviewed:
        cluster_contents = [new_clusters[id] for id in subcluster_ids if id in new_clusters]
        cluster_contents = [ing for subcluster in cluster_contents for ing in subcluster]
        print "Reviewing cluster",cluster_label+".  Contents:",", ".join(sorted(cluster_contents))
        response = ""
        while response == "":
            response = raw_input("  * Enter (v) to verify, (s) to split, (d) to disband and re-assign on next run, (r) to rename, (e) to edit, or (q) to quit:")
            if response == "v":
                clusters_reviewed.append(cluster_label)
                save_clusters_post_review(clusters_reviewed)
            elif response == "s":
                for id in subcluster_ids:
                    if id in new_clusters:
                        available_labels = set(cluster_labels.values())
                        cluster = new_clusters[id]
                        print ", ".join([str(index)+":"+ing for index,ing in zip(range(len(cluster)),cluster)])
                        response = raw_input("What name would you give this cluster? (will merge if a name is picked from above, otherwise will create a new cluster)")
                        if response in available_labels:
                            print "Adding the current ingredients to existing cluster",response
                        else:
                            print "Creating a new cluster, `"+response+"'."
                        cluster_labels[id] = response
                    save_clusters(original_clusters, new_clusters, clusters_checked, cluster_labels, verbose=False)
                response = ""
            elif response == "r":
                new_name = raw_input("  * Enter a new cluster name:")
                for id in subcluster_ids:
                    cluster_labels[id] = new_name
                save_clusters(original_clusters,new_clusters,clusters_checked,cluster_labels,verbose=False)
                clusters_reviewed.append(new_name)
                save_clusters_post_review(clusters_reviewed)
            elif response == "d":
                for id in subcluster_ids:
                    if id in new_clusters:
                        for ing in new_clusters[id]:
                            new_clusters[-1].append(ing)
                        del new_clusters[id]
                    del cluster_labels[id]
                save_clusters(original_clusters,new_clusters,clusters_checked,cluster_labels,verbose=False)
            elif response == "e":
                for id in subcluster_ids:
                    if id in new_clusters:
                        cluster = new_clusters[id]
                        print ", ".join([str(index)+":"+ing for index,ing in zip(range(len(cluster)),cluster)])
                        edit_cluster(id,cluster,new_clusters)
                save_clusters(original_clusters,new_clusters,clusters_checked,cluster_labels,verbose=False)
                response = ""
            elif response == "q":
                print "Exiting."
                sys.exit()
            else:
                response = ""

print "All revisions complete.  Printing clusters for review."

for cluster_label,subcluster_ids in sorted(inverted_cluster_labels.iteritems(),key=lambda x:x[0]):
    cluster_contents = [new_clusters[id] for id in subcluster_ids if id in new_clusters]
    cluster_contents = [ing for subcluster in cluster_contents for ing in subcluster]
    print "Reviewing cluster", cluster_label + ".  Contents:", ", ".join(sorted(cluster_contents))

print "The following ingredients were not able to be clustered:"
for ing in new_clusters[-3]:
    print "  "+ing