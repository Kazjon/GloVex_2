import pandas as pd
import re
import numpy as np
import sys
import scipy.sparse
import csv
import unicodecsv
import gensim
import itertools
from recipe_surprise_utils import pmi_scorer, load_clustering, append_clusters, pairwise_surprise, evaluate_recipe_surprise
from pprint import pprint
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
import scipy.stats
import sklearn, sklearn.cluster, sklearn.ensemble, sklearn.gaussian_process, sklearn.neural_network, sklearn.tree
from copy import copy
import hdbscan
from collections import Counter, defaultdict
import seaborn as sns
from sklearn.manifold import TSNE
import surprise
import json
import io



LIKERT_MAP = {
    'Not sure': float("nan"),
    'Not at all': 1.,
    'Somewhat': 2.,
    'Moderately': 3.,
    'Very much': 4.,
    'Extremely': 5.
}

KNOWLEDGE_LIST = {
            'chinese': ['Bamboo shoots', 'Ginger', 'Water chestnuts'],
            'mexican': ['Tortillas', 'Green chilli peppers', 'Pinto beans'],
            'italian': ['Pasta', 'Prosciutto', 'Ricotta'],
            'greek': ['Lamb', 'Feta cheese', 'Phyllo pastry'],
            'indian': ['Ghee', 'Turmeric', 'Cumin'],
            'thai': ['Fish sauce', 'Lemongrass', 'Coconut milk']
        }


def get_cuisine_fam_pref(food_cuisine_survey_df, search_str, col_str, this_col):
    # Get familiarity column
    familiar_col = re.search(search_str + '(.*) cuisine', this_col)
    if not familiar_col is None:
        # Get cuisine name and its suffix
        cuisine_name = familiar_col.group(1).lower() + col_str
        # Append cuisine name and its suffix
        food_cuisine_survey_df[cuisine_name] = food_cuisine_survey_df[this_col].map(LIKERT_MAP)

def get_recipe_surp_pref(food_cuisine_survey_df, search_str, col_str, this_col):
    surprise_col = re.search(search_str, this_col)
    if not surprise_col is None:
        # Get surprise recipe's number
        each_col_num = re.findall('\\b\\d+\\b', this_col)
        if len(each_col_num) == 0:
            surprise_recipe_num = 0
        else:
            surprise_recipe_num = each_col_num[0]
        # Name the surprise rating column
        surprise_rating_col = 'recipe' + str(surprise_recipe_num) + col_str
        # Convert the users' inputs to ints
        food_cuisine_survey_df[surprise_rating_col] = food_cuisine_survey_df[this_col].map(LIKERT_MAP)

def num_right_in(answer,cuisine):
    right_guesses = sum([i in answer for i in KNOWLEDGE_LIST[cuisine]])
    num_guesses = 1 + answer.count(",")
    wrong_guesses = num_guesses - right_guesses
    return 2 - wrong_guesses + right_guesses

def get_knowledge(food_cuisine_survey_df, knowledge_str, this_col):
    knowledge_col = re.search(knowledge_str + '(.*) cuisine', this_col)
    if not knowledge_col is None:
        cuisine_name = knowledge_col.group(1).lower()
        col_name = cuisine_name + '_knowledge'
        food_cuisine_survey_df[col_name] = food_cuisine_survey_df[this_col].apply(num_right_in,args=(cuisine_name,))


def create_users_input_dict(food_cuisine_survey_df, required_cols, scale):
    # Put the surprise scores in a DF
    users_df = food_cuisine_survey_df[required_cols]
    # Get the surprise list and put in a column
    users_df['col_list'] = (np.array(users_df.values.tolist()) / scale).tolist()
    # Create a dict with the keys as the index and the values as the surprise ratings
    return pd.Series(users_df['col_list'].values, index=users_df.index).to_dict()

def remove_highly_unsure(food_cuisine_survey_df):
    _surprise_rating_cols_ = [col for col in food_cuisine_survey_df if '_surp' in col]
    num_surp_ratings = len(_surprise_rating_cols_)
    unsure_surprise_ratings = (food_cuisine_survey_df[_surprise_rating_cols_].isna()).sum(axis=1)
    sure_users = unsure_surprise_ratings[unsure_surprise_ratings < (num_surp_ratings / 2)]
    sure_users_index = list(sure_users.index)
    return food_cuisine_survey_df.loc[sure_users_index]

# read_survey
def read_survey(food_cuisine_survey_fn, normalise_surprise=False):
    food_cuisine_survey_df = pd.read_csv(food_cuisine_survey_fn)
    print 'Total number of respondents:',len(food_cuisine_survey_df)

    # Parse users' familiarity, knowledge and preferences, recipes' surprise and preferences
    familiar_str = 'How familiar are you with '
    preference_str = 'How much do you enjoy eating '
    knowledge_str = 'Which of the following ingredients are commonly found in '
    surprise_rating_str = 'How surprising did you find this recipe?'
    surprise_preference_str = "How much do you think you'd enjoy eating this dish?"
    ingredient_rating_columns = ["Chicken", "Fish", "Beef", "Pork", "Bread", "Rice", "Pasta/noodles", "Eggs", "Milk",
                                 "Cheese", "Lettuce", "Spinach", "Potatoes", "Onions", "Tomatoes", "Strawberries",
                                 "Mangoes", "Oranges", "Apples", "Almonds"]
    for this_col in food_cuisine_survey_df.columns:
        get_cuisine_fam_pref(food_cuisine_survey_df, familiar_str, '_fam', this_col)
        get_cuisine_fam_pref(food_cuisine_survey_df, preference_str, '_pref', this_col)
        get_recipe_surp_pref(food_cuisine_survey_df, surprise_rating_str, '_surp', this_col)
        get_recipe_surp_pref(food_cuisine_survey_df, surprise_preference_str, '_pref', this_col)
        get_knowledge(food_cuisine_survey_df, knowledge_str, this_col)
        ing_match = [i in this_col for i in ingredient_rating_columns]
        if any(ing_match):
            column_name = "enjoy_"+ingredient_rating_columns[ing_match.index(True)].lower()
            food_cuisine_survey_df[column_name] = food_cuisine_survey_df[this_col].map(LIKERT_MAP)

    num_users = len(food_cuisine_survey_df)
    food_cuisine_survey_df["id"] = range(num_users)
    # Filter out users who didn't pay attention
    attention_question = 'Select only the ingredient that is Broccoli'
    food_cuisine_survey_df = food_cuisine_survey_df[food_cuisine_survey_df[attention_question] == 'Broccoli']
    # Remove highly unsure users
    food_cuisine_survey_df = remove_highly_unsure(food_cuisine_survey_df)
    print 'Number of removed respondents:', num_users - len(food_cuisine_survey_df)

    # Calculate average self-reported familiarity
    familiar_cols = [col for col in food_cuisine_survey_df if '_fam' in col]
    food_cuisine_survey_df['avg_cuisine_fam'] = food_cuisine_survey_df[familiar_cols].mean(axis=1)
    food_cuisine_survey_df['stdev_cuisine_fam'] = food_cuisine_survey_df[familiar_cols].std(axis=1)
    # Calculate average knowledge
    cuisine_knowledge_cols = [col for col in food_cuisine_survey_df if '_knowledge' in col]
    food_cuisine_survey_df['avg_cuisine_knowledge'] = food_cuisine_survey_df[cuisine_knowledge_cols].mean(axis=1)
    food_cuisine_survey_df['stdev_cuisine_knowledge'] = food_cuisine_survey_df[cuisine_knowledge_cols].std(axis=1)
    # Calculate average knowledge
    cuisine_pref_cols = [col for col in food_cuisine_survey_df if '_pref' in col]
    food_cuisine_survey_df['avg_cuisine_pref'] = food_cuisine_survey_df[cuisine_pref_cols].mean(axis=1)
    food_cuisine_survey_df['stdev_cuisine_pref'] = food_cuisine_survey_df[cuisine_pref_cols].std(axis=1)
    # Calculate average ingredient enjoyment
    ing_enjoy_cols = [col for col in food_cuisine_survey_df if 'enjoy_' in col]
    food_cuisine_survey_df['avg_ingredient_enjoyment'] = food_cuisine_survey_df[ing_enjoy_cols].mean(axis=1)
    food_cuisine_survey_df['stdev_ingredient_enjoyment'] = food_cuisine_survey_df[ing_enjoy_cols].std(axis=1)
    # Calculate average surprise ratings
    food_cuisine_survey_df["avg_surprise_rating"] = food_cuisine_survey_df[[col for col in food_cuisine_survey_df if "recipe" in col and "_surp" in col]].mean(axis=1)

    #Go through each recipe and calculate "average of all other prefs" and "average of all other surprises" columns for each user
    for i in range(16):
        other_recipe_surp_cols = [col for col in food_cuisine_survey_df if "recipe" in col and "_surp" in col and not "recipe"+str(i)+"_" in col]
        other_recipe_pref_cols = [col for col in food_cuisine_survey_df if "recipe" in col and "_pref" in col and not "recipe"+str(i)+"_" in col]
        food_cuisine_survey_df["recipe"+str(i)+"_othersurpavg"] = food_cuisine_survey_df[other_recipe_surp_cols].mean(axis=1)
        food_cuisine_survey_df["recipe"+str(i)+"_otherprefavg"] = food_cuisine_survey_df[other_recipe_pref_cols].mean(axis=1)
        food_cuisine_survey_df["recipe"+str(i)+"_othersurpstdev"] = food_cuisine_survey_df[other_recipe_surp_cols].std(axis=1)
        food_cuisine_survey_df["recipe"+str(i)+"_otherprefstdev"] = food_cuisine_survey_df[other_recipe_pref_cols].std(axis=1)

    out_df = pd.DataFrame()
    #copy across derived columns
    for col_name in food_cuisine_survey_df.columns:
        if col_name == "id" or "_" in col_name:
            out_df[col_name] = food_cuisine_survey_df[col_name]
    out_df.set_index("id",inplace=True)
    out_df["blank"] = np.zeros(len(out_df))
    return out_df,food_cuisine_survey_df

def append_recipe_specific_predictors(id,predictors,num_recipes):
    if "othersurpavg" in predictors:
        predictors.remove("othersurpavg")
        predictors.append("recipe" + str(id) + "_othersurpavg")
    if "otherprefavg" in predictors:
        predictors.remove("otherprefavg")
        predictors.append("recipe" + str(id) + "_otherprefavg")
    if "othersurpstdev" in predictors:
        predictors.remove("othersurpstdev")
        predictors.append("recipe" + str(id) + "_othersurpstdev")
    if "otherprefstdev" in predictors:
        predictors.remove("otherprefstdev")
        predictors.append("recipe" + str(id) + "_otherprefstdev")
    if "all_other_surps" in predictors:
        predictors.remove("all_other_surps")
        for other_id in range(num_recipes):
            #if other_id is not id:
            predictors.append("recipe" + str(other_id) + "_surp")
            #else:
            #    predictors.append("blank")
    if "all_other_prefs" in predictors:
        predictors.remove("all_other_prefs")
        for other_id in range(num_recipes):
            #if other_id is not id:
            predictors.append("recipe" + str(other_id) + "_pref")
            #else:
            #    predictors.append("blank")

def experiment1(mturk_recipes,mturk_recipe_ratings):
    mean_ratings = [r["surp_rating_mean"] for r in mturk_recipe_ratings]
    mean_perc_high = [r["surp_rating_perc_high"] for r in mturk_recipe_ratings]
    mean_scores_100 = [r["100%ile_surp"] for r in mturk_recipe_ratings]
    mean_scores_95 = [r["95%ile_surp"] for r in mturk_recipe_ratings]
    mean_scores_90 = [r["90%ile_surp"] for r in mturk_recipe_ratings]
    mean_scores_50 = [r["50%ile_surp"] for r in mturk_recipe_ratings]

    names = [r[0] for r in mturk_recipes]

    print "mean_ratings", zip(names, ["{:.2f}".format(p) for p in mean_ratings])
    print "mean_scores_100 correlation:", scipy.stats.pearsonr(mean_ratings, mean_scores_100)
    print "mean_scores_95 correlation:", scipy.stats.pearsonr(mean_ratings, mean_scores_95)
    print "mean_scores_90 correlation:", scipy.stats.pearsonr(mean_ratings, mean_scores_90)
    print "mean_scores_50 correlation:", scipy.stats.pearsonr(mean_ratings, mean_scores_50)


    print "mean_perc_high", zip(names, ["{:.2f}".format(p) for p in mean_perc_high])
    print "mean_scores_100 correlation:", scipy.stats.pearsonr(mean_perc_high, mean_scores_100)
    print "mean_scores_95 correlation:", scipy.stats.pearsonr(mean_perc_high, mean_scores_95)
    print "mean_scores_90 correlation:", scipy.stats.pearsonr(mean_perc_high, mean_scores_90)
    print "mean_scores_50 correlation:", scipy.stats.pearsonr(mean_perc_high, mean_scores_50)

    #plt.plot(mean_ratings,mean_scores_100,"ro")
    plt.plot(mean_ratings,[-1 * x for x in mean_scores_95],"go")
    #plt.plot([100*p for p in mean_perc_high],[-1 * x for x in mean_scores_90],"go")
    #for x, y, s in zip([100 * p for p in mean_perc_high], [-1 * x for x in mean_scores_90], [r[0] for r in mturk_recipes]):
    for x, y, s in zip(mean_ratings, [-1 * x for x in mean_scores_95], [r[0] for r in mturk_recipes]):
        plt.text(x, y, s, fontsize=12)
    #plt.plot(mean_ratings,mean_scores_50,"yo")

    plt.title("Surprise correlation (superclusters, r=0.46)")
    #plt.xlabel("% who rated surprise >= 3")
    plt.xlabel("Mean surprise rating")
    #plt.xlim([1, 3.5])
    #plt.ylim([1, 3.5])
    plt.ylabel("Surprise score (superclusters)")
    plt.show()
    #plt.plot(mean_perc_high,mean_scores_100,"r+")
    #plt.plot(mean_perc_high,mean_scores_95,"b+")
    #plt.plot(mean_ratings,mean_scores_95,"g+")
    #plt.plot(mean_perc_high,mean_scores_50,"y+")
    #plt.show()

    scaler = sklearn.preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(np.array([mean_scores_100, mean_scores_95, mean_scores_90, mean_scores_50]).T)
    #X_scaled = scaler.fit_transform(np.array([mean_scores_100]).T)
    rlr = sklearn.linear_model.LinearRegression().fit(X_scaled, mean_ratings)
    print "Linear regression R^2 on mean_ratings with all percentiles:", str(rlr.score(X_scaled, mean_ratings))
    print "  per-recipe error:", zip(names, [sklearn.metrics.mean_absolute_error(rlr.predict(np.atleast_2d(x)), [y]) for x, y in zip(X_scaled, mean_ratings)])
    plt.plot(mean_ratings, rlr.predict(X_scaled), "g+")
    for x, y, s in zip(mean_ratings, rlr.predict(X_scaled), [r[0] for r in mturk_recipes]):
        plt.text(x, y, s)
    plt.title("Actual vs predicted surprise rating")
    plt.xlabel("mean_ratings")
    plt.xlim([1, 3.5])
    plt.ylim([1, 3.5])
    plt.ylabel("predicted")
    plt.plot([1, 5], [1, 5], linewidth=0.5)
    plt.show()
    plr = sklearn.linear_model.LinearRegression().fit(X_scaled, mean_perc_high)
    print "Linear regression R^2 on mean_perc_high:", str(plr.score(X_scaled, mean_perc_high))
    print "  per-recipe error:", zip(names, [sklearn.metrics.mean_absolute_error(plr.predict(np.atleast_2d(x)), [y]) for x, y in zip(X_scaled, mean_perc_high)])
    plt.plot(mean_perc_high, plr.predict(X_scaled), "g+")
    for x, y, s in zip(mean_perc_high, plr.predict(X_scaled), [r[0] for r in mturk_recipes]):
        plt.text(x, y, s)
    plt.title("Actual vs predicted surprise %")
    plt.xlabel("mean_perc_high")
    plt.ylabel("predicted")
    plt.xlim([0.1, 0.7])
    plt.ylim([0.1, 0.7])
    plt.plot([0, 1], [0, 1], linewidth=0.5)
    plt.show()


def experiment2(parsed_survey,mturk_recipes,mturk_recipe_ratings,trials=10, c_vals=(1,), plot_scores=False):
    predictor_sets = [

        [u"othersurpavg"],

        [u"otherprefavg"],

        [u"all_other_surps"],

        [u"all_other_prefs"],

        [u"all_other_prefs", u"all_other_surps"],

        [u"otherprefstdev", u"othersurpstdev",u"otherprefavg", u"othersurpavg"],

        [u'avg_cuisine_fam', u'avg_cuisine_knowledge', u'avg_cuisine_pref', u'avg_ingredient_enjoyment'],

        #[u'avg_cuisine_fam', u'avg_cuisine_knowledge', u'avg_cuisine_pref', u'avg_ingredient_enjoyment', u"otherprefavg", u"othersurpavg"],

        [u'avg_cuisine_fam', u'avg_cuisine_knowledge', u'avg_cuisine_pref', u'avg_ingredient_enjoyment', u"otherprefavg", u"othersurpavg", u"otherprefstdev", u"othersurpstdev"],

        #[u'avg_cuisine_fam', u'avg_cuisine_knowledge', u'avg_cuisine_pref', u'avg_ingredient_enjoyment', u"otherprefavg", u"otherprefstdev", u"all_other_surps"],
        #[u'avg_cuisine_fam', u'avg_cuisine_knowledge', u'avg_cuisine_pref', u'avg_ingredient_enjoyment', u"othersurpavg", u"othersurpstdev", u"all_other_prefs"],
        [u'avg_cuisine_fam', u'avg_cuisine_knowledge', u'avg_cuisine_pref', u'avg_ingredient_enjoyment', u"all_other_surps"],
        [u'avg_cuisine_fam', u'avg_cuisine_knowledge', u'avg_cuisine_pref', u'avg_ingredient_enjoyment', u"all_other_prefs", u"all_other_surps"],

        #[u'avg_cuisine_fam', u'avg_cuisine_knowledge', u'avg_cuisine_pref', u'avg_ingredient_enjoyment', u'stdev_cuisine_fam', u'stdev_cuisine_knowledge', u'stdev_cuisine_pref',
        # u'stdev_ingredient_enjoyment', u"otherprefavg", u"othersurpavg"],

        #[u'avg_cuisine_fam', u'avg_cuisine_knowledge', u'avg_cuisine_pref', u'avg_ingredient_enjoyment', u'stdev_cuisine_fam', u'stdev_cuisine_knowledge', u'stdev_cuisine_pref',
        # u'stdev_ingredient_enjoyment', u"otherprefavg", u"othersurpavg", u"otherprefstdev", u"othersurpstdev"],

        #[u'avg_cuisine_fam', u'avg_cuisine_knowledge', u'avg_cuisine_pref', u'avg_ingredient_enjoyment', u'stdev_cuisine_fam', u'stdev_cuisine_knowledge', u'stdev_cuisine_pref',
        # u'stdev_ingredient_enjoyment', u"otherprefavg", u"all_other_surps", u"otherprefstdev"],

        #[u'chinese_pref', u'chinese_fam', u'chinese_knowledge', u'mexican_pref', u'mexican_fam', u'mexican_knowledge', u'italian_pref', u'italian_fam',
        # u'italian_knowledge', u'greek_pref', u'greek_fam', u'greek_knowledge', u'indian_pref', u'indian_fam', u'indian_knowledge', u'thai_pref',
        # u'thai_fam', u'thai_knowledge', u'avg_cuisine_fam', u'avg_cuisine_knowledge', u'avg_cuisine_pref', u'avg_ingredient_enjoyment',u'stdev_cuisine_fam',
        # u'stdev_cuisine_knowledge', u'stdev_cuisine_pref', u'stdev_ingredient_enjoyment', u"otherprefavg", u"othersurpavg"],

        #[u'enjoy_chicken', u'enjoy_fish', u'enjoy_beef', u'enjoy_pork', u'enjoy_bread', u'enjoy_rice',
        # u'enjoy_pasta/noodles', u'enjoy_eggs', u'enjoy_milk', u'enjoy_cheese', u'enjoy_lettuce', u'enjoy_spinach', u'enjoy_potatoes', u'enjoy_onions',
        # u'enjoy_tomatoes', u'enjoy_strawberries', u'enjoy_mangoes', u'enjoy_oranges', u'enjoy_apples', u'enjoy_almonds', u'avg_cuisine_fam', u'avg_cuisine_knowledge',
        # u'avg_cuisine_pref', u'avg_ingredient_enjoyment',u'stdev_cuisine_fam', u'stdev_cuisine_knowledge', u'stdev_cuisine_pref', u'stdev_ingredient_enjoyment',
        # u"otherprefavg", u"othersurpavg"],

        #[u'chinese_pref', u'chinese_fam', u'chinese_knowledge', u'mexican_pref', u'mexican_fam', u'mexican_knowledge', u'italian_pref', u'italian_fam',
        # u'italian_knowledge', u'greek_pref', u'greek_fam', u'greek_knowledge', u'indian_pref', u'indian_fam', u'indian_knowledge', u'thai_pref',
        # u'thai_fam', u'thai_knowledge', u'enjoy_chicken', u'enjoy_fish', u'enjoy_beef', u'enjoy_pork', u'enjoy_bread', u'enjoy_rice',
        # u'enjoy_pasta/noodles', u'enjoy_eggs', u'enjoy_milk', u'enjoy_cheese', u'enjoy_lettuce', u'enjoy_spinach', u'enjoy_potatoes', u'enjoy_onions',
        # u'enjoy_tomatoes', u'enjoy_strawberries', u'enjoy_mangoes', u'enjoy_oranges', u'enjoy_apples', u'enjoy_almonds',u'avg_cuisine_fam', u'avg_cuisine_knowledge',
        # u'avg_cuisine_pref', u'avg_ingredient_enjoyment']
    ]

    for c in c_vals:
        for predictors in predictor_sets:
            print
            print
            print "C="+str(c)+", Predictors:",predictors
            deltas = []
            error_reductions = []
            recalls = []
            precisions = []
            f1s = []
            for id,r in enumerate(mturk_recipes):
                r_predictors = copy(predictors)
                title = r[0]
                ings = r[1]
                surp_ratings = mturk_recipe_ratings[id]["surp_ratings"]
                nan_mask = ~np.isnan(np.array(surp_ratings))
                high_surp_ratings = np.array(surp_ratings)[nan_mask]>=3

                base_score = max(np.mean(high_surp_ratings), 1 - np.mean(high_surp_ratings))
                #base_score = 0.5

                append_recipe_specific_predictors(id,r_predictors,len(mturk_recipes))
                X = parsed_survey[r_predictors][nan_mask].fillna(0)
                if "recipe" + str(id) + "_surp" in X.columns:
                    X["recipe" + str(id) + "_surp"] = mturk_recipe_ratings[id]["surp_rating_mean"]
                if "recipe" + str(id) + "_pref" in X.columns:
                    X["recipe" + str(id) + "_pref"] = mturk_recipe_ratings[id]["pref_rating_mean"]

                scaler = sklearn.preprocessing.StandardScaler()
                X_scaled = scaler.fit_transform(X)

                r_accuracies = []
                r_recalls = []
                r_precisions = []
                r_f1s = []
                for i in range(trials):
                    #clf = sklearn.linear_model.LogisticRegression(tol=10e-6, C=c)#, class_weight="balanced")
                    #clf = sklearn.svm.SVC(C=c)
                    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
                    kf = sklearn.model_selection.StratifiedKFold(n_splits=5,shuffle=True)
                    for train_indices,test_indices in kf.split(X_scaled,high_surp_ratings):
                        X_train = X_scaled[train_indices]
                        X_test = X_scaled[test_indices]
                        y_train = high_surp_ratings[train_indices]
                        y_test = high_surp_ratings[test_indices]
                        clf.fit(X_train,y_train)
                        test_pred = clf.predict(X_test)
                        r_accuracies.append(sklearn.metrics.accuracy_score(y_test,test_pred))
                        r_recalls.append(sklearn.metrics.recall_score(y_test,test_pred))
                        r_precisions.append(sklearn.metrics.precision_score(y_test,test_pred))
                        r_f1s.append(sklearn.metrics.f1_score(y_test,test_pred))
                score = sum(r_accuracies)/len(r_accuracies)
                recall = sum(r_recalls)/len(r_recalls)
                recalls.append(recall*100)
                precision = sum(r_precisions)/len(r_precisions)
                precisions.append(precision*100)
                f1 = sum(r_f1s)/len(r_f1s)
                f1s.append(f1*100)
                delta = score - base_score
                deltas.append(delta*100)
                error_reduction = delta/(1-base_score)
                error_reductions.append(error_reduction*100)

                print title+":",
                print "{:.2f}".format(base_score*100),
                print "{:.2f}".format(score*100),
                print "{:+.2f}".format(delta*100),
                print "({:+.2f}%)".format(error_reduction*100),
                print "recall={:.2f}%".format(recall*100),
                print "precision={:.2f}%".format(precision*100),
                print "f1={:.2f}%".format(f1*100)
            print "AVERAGE IMPROVEMENT:",
            print "{:+.2f}".format(sum(deltas)/len(deltas)),
            print "({:+.2f}%)".format(sum(error_reductions)/len(error_reductions)),
            print "recall={:.2f}%".format(sum(recalls)/len(recalls)),
            print "precision={:.2f}%".format(sum(precisions)/len(precisions)),
            print "f1={:.2f}%".format(sum(f1s)/len(f1s))
            if plot_scores:
                xs = [mturk_recipe_ratings[i]["surp_rating_perc_high"] for i in range(len(mturk_recipes))]
                ys = recalls
                plt.plot(xs,ys,"b+")
                for x,y,s in zip(xs,ys,[r[0] for r in mturk_recipes]):
                    plt.text(x,y,s)
                plt.title("Recall vs surp% for "+", ".join(r_predictors))
                plt.show()
                ys = precisions
                plt.plot(xs,ys,"r+")
                for x,y,s in zip(xs,ys,[r[0] for r in mturk_recipes]):
                    plt.text(x,y,s)
                plt.title("Precision vs surp% for "+", ".join(r_predictors))
                plt.show()
                ys = error_reductions
                plt.plot(xs,ys,"g+")
                for x,y,s in zip(xs,ys,[r[0] for r in mturk_recipes]):
                    plt.text(x,y,s)
                plt.title("Error% vs surp% for "+", ".join(r_predictors))
                plt.show()

def experiment3(parsed_survey,mturk_recipes,mturk_recipe_ratings, c_vals=(1,),trials=1,discretise=True, plot=True, use_heldout=False, hold_users=True, n_splits=5):
    cluster = False #This little experiment turned out not to be necessary
    if cluster:
        recipe_mean_surps = [i["surp_rating_mean"] for i in mturk_recipe_ratings]
        for min_cluster_size in [3,4,5,8,10]:
            surp_array = np.array([i["surp_ratings"] for i in mturk_recipe_ratings]).T
            surp_array_embedded = TSNE().fit_transform(surp_array)
            print "Constructed TSNE embedding."
            clusterer = sklearn.cluster.MiniBatchKMeans(n_clusters=min_cluster_size)
            #clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_cluster_size, core_dist_n_jobs=8, cluster_selection_method="eom")
            clusterer.fit(surp_array)
            print "Fit clusterer."
            #clusterer.condensed_tree_.plot(cmap='viridis', colorbar=True)
            #plt.show()

            num_clusters = len(set(clusterer.labels_)) - 1
            num_outliers = sum(clusterer.labels_ == -1)

            print "  Found", num_clusters, "clusters with", num_outliers, "outliers, for", num_clusters + num_outliers, "total features."
            cluster_sizes = Counter(list(clusterer.labels_))
            clusters_to_print = 500
            for cluster, cluster_size in cluster_sizes.most_common(clusters_to_print + 1):
                    cluster_indices = np.where(clusterer.labels_ == cluster)[0]
                    print cluster, cluster_size
                    print ", ".join(["{:+.2f}".format(cm-m) for cm,m in zip(np.mean(surp_array[cluster_indices, :],axis=0),recipe_mean_surps)])
            color_palette = sns.color_palette('deep', len(set(clusterer.labels_)))
            color_palette[-1] = (0.5, 0.5, 0.5)
            cluster_colours = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in clusterer.labels_]
            cluster_data = pd.DataFrame(surp_array_embedded, columns=["x", "y"])
            plt.scatter(x=cluster_data["x"], y=cluster_data["y"], c=cluster_colours, alpha=0.3)
            plt.show()

    predictor_sets = [

        #[u"surp_score"],
        #[u"surp_score", u'avg_cuisine_fam', u'avg_cuisine_knowledge', u'avg_cuisine_pref', u'avg_ingredient_enjoyment'],
        #[u"surp_score", u'avg_cuisine_fam', u'avg_cuisine_knowledge', u'avg_cuisine_pref', u'avg_ingredient_enjoyment', u'stdev_cuisine_fam', u'stdev_cuisine_knowledge',
        # u'stdev_cuisine_pref',u'stdev_ingredient_enjoyment'],
        #[u"surp_score", u"othersurpavg", u"othersurpstdev"],
        #[u"surp_score", u"othersurpavg", u"otherprefavg", u"otherprefstdev", u"othersurpstdev"],
        [u"surp_score", u"all_other_surps"],
        #[u"surp_score", u"all_other_surps", u"all_other_prefs"],
        #[u'avg_cuisine_fam', u"othersurpavg"],
        #[u'avg_cuisine_fam', u'avg_cuisine_knowledge', u"othersurpavg"],
        #[u'avg_cuisine_fam', u'avg_cuisine_knowledge', u'avg_cuisine_pref', u"othersurpavg"],
        #[u'avg_cuisine_fam', u'avg_cuisine_knowledge', u'avg_cuisine_pref', u'avg_ingredient_enjoyment', u"othersurpavg"],
        #[u'avg_cuisine_knowledge', u'avg_cuisine_pref', u'avg_ingredient_enjoyment', u"othersurpavg", u"otherprefavg"],
        #[u'avg_cuisine_knowledge', u'avg_cuisine_pref', u'avg_ingredient_enjoyment', u"othersurpavg", u"otherprefavg", u"otherprefstdev", u"othersurpstdev"],
        #[u'avg_cuisine_fam', u'avg_cuisine_knowledge', u'avg_cuisine_pref', u'avg_ingredient_enjoyment', u"otherprefavg", u"othersurpavg"],
        #[u'avg_cuisine_fam', u'avg_cuisine_knowledge', u'avg_cuisine_pref', u'avg_ingredient_enjoyment', u"otherprefavg", u"otherprefstdev", u"othersurpavg", u"othersurpstdev"],
        #[u'avg_cuisine_fam', u'avg_cuisine_knowledge', u'avg_cuisine_pref', u'avg_ingredient_enjoyment', u"otherprefavg", u"otherprefstdev", u"all_other_surps"],
        #[u'avg_cuisine_fam', u'avg_cuisine_knowledge', u'avg_cuisine_pref', u'avg_ingredient_enjoyment', u"otherprefavg", u"otherprefstdev", u"othersurpavg", u"othersurpstdev", u"all_other_surps"]
    ]

    #comparison_sets = {"familiarity":set(np.where(parsed_survey["avg_cuisine_fam"]>=parsed_survey["avg_cuisine_fam"].median())[0]),
    #                   "knowledge": set(np.where(parsed_survey["avg_cuisine_knowledge"] >= parsed_survey["avg_cuisine_knowledge"].median())[0]),
    #                   "preference": set(np.where(parsed_survey["avg_cuisine_pref"] >= parsed_survey["avg_cuisine_pref"].median())[0]),
    #                   "enjoyment": set(np.where(parsed_survey["avg_ingredient_enjoyment"] >= parsed_survey["avg_ingredient_enjoyment"].median())[0]),
    #                   "surp_ratings": set(np.where(parsed_survey["avg_surprise_rating"] >= parsed_survey["avg_surprise_rating"].median())[0])
    #                   }


    fam_median = parsed_survey["avg_cuisine_fam"].median()
    know_median = parsed_survey["avg_cuisine_knowledge"].median()
    pref_median = parsed_survey["avg_cuisine_pref"].median()
    enjoy_median = parsed_survey["avg_ingredient_enjoyment"].median()
    surp_avg_median = parsed_survey["avg_surprise_rating"].median()

    for c in c_vals:
        for predictors in predictor_sets:
            print
            print
            print "C="+str(c)+", Predictors:",predictors
            deltas = []
            if discretise:
                error_reductions = []
                recalls = []
                precisions = []
                f1s = []
            else:
                maes = []
                expvars = []
                r2s = []
            X = []
            y = []
            recipe_indices = []
            user_perc_high = np.zeros(len(parsed_survey),dtype="int64")

            comparison_sets = {"familiarity": set(),
                               "knowledge": set(),
                               "preference": set(),
                               "enjoyment": set(),
                               "surp_ratings": set()
                               }

            X_to_user_index = []

            for id,r in enumerate(mturk_recipes):
                r_predictors = copy(predictors)
                recipe_indices.append(set())

                append_recipe_specific_predictors(id,r_predictors,len(mturk_recipes))

                surp_ratings = mturk_recipe_ratings[id]["surp_ratings"]
                if discretise:
                    nan_mask = ~np.isnan(np.array(surp_ratings))
                    high_surp_ratings = (np.array(surp_ratings)>=3)
                    user_perc_high += high_surp_ratings
                else:
                    high_surp_ratings = surp_ratings
                rX = parsed_survey[[p for p in r_predictors if not p == "surp_score"]].fillna(0)
                if "recipe" + str(id) + "_surp" in rX.columns:
                    rX["recipe" + str(id) + "_surp"] = mturk_recipe_ratings[id]["surp_rating_mean"]
                if "recipe" + str(id) + "_pref" in rX.columns:
                    rX["recipe" + str(id) + "_pref"] = mturk_recipe_ratings[id]["pref_rating_mean"]

                for idx,row,score,surp,isnt_nan in zip(range(len(high_surp_ratings)),
                                                       rX.values,
                                                       np.repeat(mturk_recipe_ratings[id]["90%ile_surp"],len(high_surp_ratings)),
                                                       high_surp_ratings,
                                                       nan_mask
                                                       ):
                    if isnt_nan:
                        recipe_indices[-1].add(len(X))
                        if parsed_survey["avg_cuisine_fam"]._values[idx]>fam_median:
                            comparison_sets["familiarity"].add(len(X))
                        if parsed_survey["avg_cuisine_knowledge"]._values[idx]>know_median:
                            comparison_sets["knowledge"].add(len(X))
                        if parsed_survey["avg_cuisine_pref"]._values[idx]>pref_median:
                            comparison_sets["preference"].add(len(X))
                        if parsed_survey["avg_ingredient_enjoyment"]._values[idx]>enjoy_median:
                            comparison_sets["enjoyment"].add(len(X))
                        if parsed_survey["avg_surprise_rating"]._values[idx]>surp_avg_median:
                            comparison_sets["surp_ratings"].add(len(X))
                        X_to_user_index.append(idx)
                        if "surp_score" in r_predictors:
                            X.append(list(row)+[score])
                        else:
                            X.append(list(row))
                        y.append(surp)




            scaler = sklearn.preprocessing.StandardScaler()
            X_scaled = scaler.fit_transform(X)
            num_samples = float(len(y))
            y = np.array(y)
            test_predictions = {"truth":[],"estimate":[],"user_ids":[]}
            if discretise:
                num_high = np.sum(y)
                #This is the score you get for always picking "no" on recipes that surprise <50% of people and always picking "yes" on recipes that surprise >50% of people.
                base_score = 0.6617 #max(num_high/num_samples,1-(num_high/num_samples))
                p_accuracies = []
                p_recalls = []
                p_precisions = []
                p_f1s = []
                p_userprofiles = np.zeros(len(parsed_survey))
                p_trueposprofiles = np.zeros(len(parsed_survey))
                recipe_accuracies = [[] for i in range(len(mturk_recipes))]
                recipe_recalls = [[] for i in range(len(mturk_recipes))]
                recipe_precisions = [[] for i in range(len(mturk_recipes))]
                recipe_f1s = [[] for i in range(len(mturk_recipes))]
                comparison_accuracies = {k:([],[]) for k,v in comparison_sets.iteritems()}
                comparison_recalls = {k:([],[]) for k,v in comparison_sets.iteritems()}
                comparison_precisions = {k:([],[]) for k,v in comparison_sets.iteritems()}
                comparison_f1s = {k:([],[]) for k,v in comparison_sets.iteritems()}
            else:
                base_score = np.var(y)
                p_mses = []
                p_maes = []
                p_expvars = []
                p_r2s = []
                recipe_mses = [[] for i in range(len(mturk_recipes))]
                recipe_maes = [[] for i in range(len(mturk_recipes))]
                recipe_expvars = [[] for i in range(len(mturk_recipes))]
                recipe_r2s = [[] for i in range(len(mturk_recipes))]
            for i in range(trials):
                if use_heldout:
                    if hold_users:
                        user_train_indices, user_test_indices = sklearn.model_selection.train_test_split(range(len(parsed_survey)), test_size=0.1)
                        user_train_indices = set(user_train_indices)
                        user_test_indices = set(user_test_indices)
                        train_indices = [i for i in range(len(X_scaled)) if X_to_user_index[i] in user_train_indices]
                        test_indices = [i for i in range(len(X_scaled)) if X_to_user_index[i] in user_test_indices]
                    else:
                        train_indices,test_indices = sklearn.model_selection.train_test_split(range(len(X_scaled)), test_size=0.2)
                    splits = [(train_indices,test_indices)]
                else:
                    if hold_users:
                        kf = sklearn.model_selection.KFold(n_splits=n_splits, shuffle=True)
                        user_splits = [set(s) for _,s in kf.split(np.zeros(len(parsed_survey)),range(len(parsed_survey)))]
                        splits = []
                        for user_split in user_splits:
                            splits.append(([i for i in range(len(X_scaled)) if X_to_user_index[i] not in user_split],[i for i in range(len(X_scaled)) if X_to_user_index[i] in user_split]))
                    else:
                        kf = sklearn.model_selection.StratifiedKFold(n_splits=n_splits,shuffle=True)
                        splits = kf.split(X_scaled,y)
                for train_indices,test_indices in splits:
                    if discretise:
                        #predictor = sklearn.linear_model.LogisticRegression(tol=10e-6, C=c)
                        #predictor = sklearn.gaussian_process.GaussianProcessClassifier()
                        #predictor = sklearn.svm.SVC(C=c)
                        predictor = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(64,2), max_iter=10000, early_stopping=True, n_iter_no_change=1000,
                                                                         alpha=1/float(c))
                    else:
                        #predictor = sklearn.linear_model.Ridge(alpha=1./c)
                        predictor = sklearn.svm.SVR(C=c)
                    X_train = X_scaled[train_indices]
                    X_test = X_scaled[test_indices]
                    y_train = y[train_indices]
                    y_test = y[test_indices]
                    predictor.fit(X_train,y_train)
                    test_pred = predictor.predict(X_test)
                    test_predictions["truth"].append(y_test)
                    test_predictions["estimate"].append(test_pred)
                    test_predictions["user_ids"].append([X_to_user_index[i] for i in test_indices])
                    if discretise:
                        p_accuracies.append(sklearn.metrics.accuracy_score(y_test,test_pred))
                        p_recalls.append(sklearn.metrics.recall_score(y_test,test_pred))
                        p_precisions.append(sklearn.metrics.precision_score(y_test,test_pred))
                        p_f1s.append(sklearn.metrics.f1_score(y_test,test_pred))
                        for cname,c_indices in comparison_sets.iteritems():
                            cX_test = X_scaled[[p for p in test_indices if p in c_indices]]
                            cy_test = y[[p for p in test_indices if p in c_indices]]
                            ctest_pred = predictor.predict(cX_test)
                            comparison_accuracies[cname][0].append(sklearn.metrics.accuracy_score(cy_test, ctest_pred))
                            comparison_recalls[cname][0].append(sklearn.metrics.recall_score(cy_test, ctest_pred))
                            comparison_precisions[cname][0].append(sklearn.metrics.precision_score(cy_test, ctest_pred))
                            comparison_f1s[cname][0].append(sklearn.metrics.f1_score(cy_test, ctest_pred))
                            ncX_test = X_scaled[[p for p in test_indices if p not in c_indices]]
                            ncy_test = y[[p for p in test_indices if p not in c_indices]]
                            nctest_pred = predictor.predict(ncX_test)
                            comparison_accuracies[cname][1].append(sklearn.metrics.accuracy_score(ncy_test, nctest_pred))
                            comparison_recalls[cname][1].append(sklearn.metrics.recall_score(ncy_test, nctest_pred))
                            comparison_precisions[cname][1].append(sklearn.metrics.precision_score(ncy_test, nctest_pred))
                            comparison_f1s[cname][1].append(sklearn.metrics.f1_score(ncy_test, nctest_pred))
                    else:
                        p_mses.append(sklearn.metrics.mean_squared_error(y_test,test_pred))
                        p_maes.append(sklearn.metrics.mean_absolute_error(y_test,test_pred))
                        p_expvars.append(sklearn.metrics.explained_variance_score(y_test,test_pred))
                        p_r2s.append(sklearn.metrics.r2_score(y_test,test_pred))
                    for id,r_indices in enumerate(recipe_indices):
                        rtest_indices = [p for p in test_indices if p in r_indices]
                        rX_test = X_scaled[rtest_indices]
                        ry_test = y[rtest_indices]
                        rtest_pred = predictor.predict(rX_test)
                        if discretise:
                            recipe_accuracies[id].append(sklearn.metrics.accuracy_score(ry_test,rtest_pred))
                            recipe_recalls[id].append(sklearn.metrics.recall_score(ry_test,rtest_pred))
                            recipe_precisions[id].append(sklearn.metrics.precision_score(ry_test,rtest_pred))
                            recipe_f1s[id].append(sklearn.metrics.f1_score(ry_test,rtest_pred))
                            for index,pred,truth in zip(rtest_indices,rtest_pred,ry_test):
                                p_userprofiles[X_to_user_index[index]]+=int(pred==truth)
                                p_trueposprofiles[X_to_user_index[index]]+=int(pred==True and truth==True)
                        else:
                            recipe_mses[id].append(sklearn.metrics.mean_squared_error(ry_test,rtest_pred))
                            recipe_maes[id].append(sklearn.metrics.mean_absolute_error(ry_test,rtest_pred))
                            recipe_expvars[id].append(sklearn.metrics.explained_variance_score(ry_test,rtest_pred))
                            recipe_r2s[id].append(sklearn.metrics.r2_score(ry_test,rtest_pred))

            if discretise:
                p_userprofiles /= (len(mturk_recipes)*trials)
                score = sum(p_accuracies)/len(p_accuracies)
                recall = sum(p_recalls)/len(p_recalls)
                recalls.append(recall*100)
                precision = sum(p_precisions)/len(p_precisions)
                precisions.append(precision*100)
                f1 = sum(p_f1s)/len(p_f1s)
                f1s.append(f1*100)
                delta = score - base_score
                deltas.append(delta*100)
                error_reduction = delta/(1-base_score)
                error_reductions.append(error_reduction*100)

                print "Experiment 3 scores (classification):",
                print "{:.2f}".format(base_score*100),
                print "{:.2f}".format(score*100),
                print "{:+.2f}".format(delta*100),
                print "({:+.2f}%)".format(error_reduction*100),
                print "recall={:.2f}%".format(recall*100),
                print "precision={:.2f}%".format(precision*100),
                print "f1={:.2f}%".format(f1*100)

                print "Per recipe scores:"
                for name,r_accs,r_recs,r_precs,r_f1s in zip([r[0] for r in mturk_recipes],recipe_accuracies,recipe_recalls,recipe_precisions,recipe_f1s):
                    r_rec = sum(r_recs)/len(r_recs)
                    r_acc = sum(r_accs)/len(r_accs)
                    r_prec = sum(r_precs)/len(r_precs)
                    r_f1 = sum(r_f1s)/len(r_f1s)
                    print "  "+name+": accuracy="+"{:.2f}".format(r_acc*100)+"  recall="+"{:.2f}".format(r_rec*100)+"  precision="+"{:.2f}".format(r_prec*100)+"  f1="+"{:.2f}".format(r_f1*100)

                if plot:
                    plt.bar(range(len(mturk_recipes)+1),np.bincount(user_perc_high).astype(float)/len(user_perc_high))
                    plt.plot(user_perc_high,p_userprofiles,"go",alpha=0.1,markersize=4)
                    plt.show()

                    plt.hist(p_userprofiles,bins=10)
                    plt.show()
                    plt.hist([(float(tp)/float(p))/float(trials) for tp,p in zip(p_trueposprofiles,user_perc_high) if p > 0],bins=100)
                    plt.show()

                    xs = [mturk_recipe_ratings[i]["surp_rating_perc_high"] for i in range(len(mturk_recipes))]
                    ys = [sum(r)/len(r) for r in recipe_recalls]
                    plt.plot(xs,ys,"b+")
                    for x,y,s in zip(xs,ys,[r[0] for r in mturk_recipes]):
                        plt.text(x,y,s)
                    plt.title("Recall vs surp% for "+", ".join(r_predictors))
                    plt.show()
                    ys = [sum(r)/len(r) for r in recipe_precisions]
                    plt.plot(xs,ys,"r+")
                    for x,y,s in zip(xs,ys,[r[0] for r in mturk_recipes]):
                        plt.text(x,y,s)
                    plt.title("Precision vs surp% for "+", ".join(r_predictors))
                    plt.show()
                    ys = [sum(r)/len(r) for r in recipe_accuracies]
                    plt.plot(xs,ys,"g+")
                    for x,y,s in zip(xs,ys,[r[0] for r in mturk_recipes]):
                        plt.text(x,y,s)
                    plt.title("Accuracy vs surp% for "+", ".join(r_predictors))
                    plt.show()

                print "Scores on comparison sets:"
                for cname in comparison_sets.keys():
                    acc_above = sum(comparison_accuracies[cname][0])/len(comparison_accuracies[cname][0])
                    acc_above_std = np.std(comparison_accuracies[cname][0])
                    acc_below = sum(comparison_accuracies[cname][1])/len(comparison_accuracies[cname][1])
                    acc_below_std = np.std(comparison_accuracies[cname][1])
                    rec_above = sum(comparison_recalls[cname][0])/len(comparison_recalls[cname][0])
                    rec_above_std = np.std(comparison_recalls[cname][0])
                    rec_below = sum(comparison_recalls[cname][1])/len(comparison_recalls[cname][1])
                    rec_below_std = np.std(comparison_recalls[cname][1])
                    prec_above = sum(comparison_precisions[cname][0])/len(comparison_precisions[cname][0])
                    prec_above_std = np.std(comparison_precisions[cname][0])
                    prec_below = sum(comparison_precisions[cname][1])/len(comparison_precisions[cname][1])
                    prec_below_std = np.std(comparison_precisions[cname][1])
                    f1_above = sum(comparison_f1s[cname][0])/len(comparison_f1s[cname][0])
                    f1_above_std = np.std(comparison_f1s[cname][0])
                    f1_below = sum(comparison_f1s[cname][1])/len(comparison_f1s[cname][1])
                    f1_below_std = np.std(comparison_f1s[cname][1])
                    print "  "+cname+": accuracy = "+"{:.2f}".format(acc_above)+"(+/-{:.2f})/".format(acc_above_std)+"{:.2f}".format(acc_below)+"(+/-{:.2f}),".format(acc_below_std),
                    print "recall = "+"{:.2f}".format(rec_above)+"(+/-{:.2f})/".format(rec_above_std)+"{:.2f}".format(rec_below)+"(+/-{:.2f}),".format(rec_below_std),
                    print "precision = "+"{:.2f}".format(prec_above)+"(+/-{:.2f})/".format(prec_above_std)+"{:.2f}".format(prec_below)+"(+/-{:.2f}),".format(prec_below_std),
                    print "f1 = "+"{:.2f}".format(f1_above)+"(+/-{:.2f})/".format(f1_above_std)+"{:.2f}".format(f1_below)+"(+/-{:.2f}),".format(f1_below_std)

                print "Recommender performance:"
                rec_style_predictions = [(uid, 0, true_r, est, 0) for uid, true_r, est in zip(itertools.chain.from_iterable(test_predictions["user_ids"]),
                                                                                              itertools.chain.from_iterable(test_predictions["truth"]),
                                                                                              itertools.chain.from_iterable(test_predictions["estimate"]))]
                precisions, recalls = precision_recall_at_k_binary(rec_style_predictions, k=16)
                print "  recsys_precision:",sum(prec for prec in precisions.values()) / len(precisions)
                print "  recsys_recall:",sum(rec for rec in recalls.values()) / len(recalls)

            else:
                score = sum(p_mses) / len(p_mses)
                mae = sum(p_maes) / len(p_maes)
                maes.append(mae)
                expvar = sum(p_expvars) / len(p_expvars)
                expvars.append(expvar)
                r2 = sum(p_r2s) / len(p_r2s)
                r2s.append(r2)
                delta = score - base_score
                deltas.append(delta)

                print "Experiment 3 scores (regression):",
                print "{:.2f}".format(base_score),
                print "{:.2f}".format(score),
                print "mae={:.2f}".format(mae),
                print "expvar={:.2f}%".format(expvar*100),
                print "r2={:.2f}".format(r2),

                print "Per recipe scores:"
                for name, r_mses, r_maes, r_evs, r_r2s in zip([r[0] for r in mturk_recipes], recipe_mses, recipe_maes, recipe_expvars, recipe_r2s):
                    r_mse = float(sum(r_mses)) / len(r_mses)
                    r_mae = float(sum(r_maes)) / len(r_maes)
                    r_ev = sum(r_evs) / len(r_evs)
                    r_r2 = sum(r_r2s) / len(r_r2s)
                    print "  " + name + ": mse=" + "{:.2f}".format(r_mse) + "  mae=" + "{:.2f}".format(r_mae) + "  expvar=" + "{:.2f}%".format(
                        r_ev * 100) + "  r2=" + "{:.2f}".format(r_r2)

def experiment4(parsed_survey,mturk_recipes,mturk_recipe_ratings, plot=False, hold_users=False):
    print
    print
    reader = surprise.Reader(rating_scale=(1,5))

    train_users, test_users = sklearn.model_selection.train_test_split(range(len(parsed_survey)),test_size=0.2)
    train_triplets = {'itemID': [], 'userID': [], 'rating': []}
    test_triplets = {'itemID': [], 'userID': [], 'rating': []}
    for u in range(len(parsed_survey)):
        for r,recipe in enumerate(mturk_recipe_ratings):
            if not np.isnan(recipe["surp_ratings"][u]):
                if u in train_users:
                    train_triplets["itemID"].append(r)
                    train_triplets["userID"].append(u)
                    train_triplets["rating"].append(recipe["surp_ratings"][u])
                else:
                    test_triplets["itemID"].append(r)
                    test_triplets["userID"].append(u)
                    test_triplets["rating"].append(recipe["surp_ratings"][u])

    train_triplets = pd.DataFrame(train_triplets)
    test_triplets = pd.DataFrame(test_triplets)
    train_data = surprise.Dataset.load_from_df(train_triplets[["userID", "itemID", "rating"]],reader)
    test_data = surprise.Dataset.load_from_df(test_triplets[["userID", "itemID", "rating"]],reader)

    algo = surprise.SVD(n_epochs=250)

    full_trainset = train_data.build_full_trainset()
    algo = surprise.SVD(n_epochs=250)
    algo.fit(full_trainset)
    predictions = algo.test(full_trainset.build_testset())
    print 'Biased accuracy on whole dataset: ',
    surprise.accuracy.mae(predictions)

    precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=3)
    print " precision:",sum(prec for prec in precisions.values()) / len(precisions)
    print "  recall:",sum(rec for rec in recalls.values()) / len(recalls)
    print "  accuracy:",float(sum([round(pred)==truth for _, _, truth, pred, _ in predictions]))/len(predictions)

    trainset, valset = surprise.model_selection.train_test_split(train_data, test_size=.25)
    algo.fit(trainset)
    predictions = algo.test(valset)
    print 'Unbiased accuracy on unseen pairs: ',
    surprise.accuracy.mae(predictions)

    precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=3)
    print " precision:",sum(prec for prec in precisions.values()) / len(precisions)
    print "  recall:",sum(rec for rec in recalls.values()) / len(recalls)
    print "  accuracy:",float(sum([round(pred)==truth for _, _, truth, pred, _ in predictions]))/len(predictions)

    testset = train_data.construct_testset(test_data.raw_ratings)
    predictions = algo.test(testset)
    print 'Unbiased accuracy on unseen users: ',
    surprise.accuracy.mae(predictions)

    precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=3)
    print " precision:",sum(prec for prec in precisions.values()) / len(precisions)
    print "  recall:",sum(rec for rec in recalls.values()) / len(recalls)
    print "  accuracy:",float(sum([round(pred)==truth for _, _, truth, pred, _ in predictions]))/len(predictions)

    algo = surprise.prediction_algorithms.random_pred.NormalPredictor()
    algo.fit(trainset)
    predictions = algo.test(testset)
    print 'Accuracy of a dummy distribution-based predictor: ',
    surprise.accuracy.mae(predictions)

    precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=3)
    print " precision:",sum(prec for prec in precisions.values()) / len(precisions)
    print "  recall:",sum(rec for rec in recalls.values()) / len(recalls)
    print "  accuracy:",float(sum([round(pred)==truth for _, _, truth, pred, _ in predictions]))/len(predictions)

#hoiked from https://surprise.readthedocs.io/en/stable/FAQ.html
def precision_recall_at_k(predictions, k=10, threshold=3.5):
    '''Return precision and recall at k metrics for each user.'''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()

    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((round(est) >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = float(sum(((true_r >= threshold) and (round(est) >= threshold))
                              for (est, true_r) in user_ratings[:k]))

        # Precision@K: Proportion of recommended items that are surprising
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are surprising
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls

#hoiked from https://surprise.readthedocs.io/en/stable/FAQ.html
def precision_recall_at_k_binary(predictions, k=10):
    '''Return precision and recall at k metrics for each user.'''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()

    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of surprising items
        n_surp = sum(true_r for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum(est for (est, _) in user_ratings[:k])

        # Number of surprising and recommended items in top k
        n_surp_and_rec_k = float(sum((true_r and est) for (est, true_r) in user_ratings[:k]))

        # Precision@K: Proportion of recommended items that are surprising
        precisions[uid] = n_surp_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of surprising items that are recommended
        recalls[uid] = n_surp_and_rec_k / n_surp if n_surp != 0 else 1

    return precisions, recalls


def surprise_from_recipe(recipe):
    try:
        return recipe["surprise"][0][2]
    except:
        return float("inf")


def evaluate_dataset_surprise(recipes, clustering, cluster_labels, ft_vocab, ft_model, num_clusters, supercluster_names, observed_pairs, ingredient_pair_surprises,
                                 feature_freqs, feature_coocs, use_ings=False, use_clusters=True, use_superclusters=False):
    for i,recipe in enumerate(recipes.values()):
        recipe["surprise"] = evaluate_recipe_surprise(recipe["ingredients"], clustering, cluster_labels, ft_vocab, ft_model, num_clusters, supercluster_names, observed_pairs, ingredient_pair_surprises,
                                 feature_freqs, feature_coocs, use_ings=use_ings, use_clusters=use_clusters, use_superclusters=use_superclusters)
        if i and i%100 == 0:
                print ".",
    print
    print
    recipe_list = sorted(recipes.values(),key=surprise_from_recipe)
    for recipe in recipe_list[:100]:
        print recipe["title"],recipe["ingredients"]
        for surprise in recipe["surprise"][:3]:
            print "  "+str(surprise)




if __name__ == "__main__":
    surprise_file = sys.argv[1]
    feature_freqs_file = sys.argv[2]
    feature_cooc_file = sys.argv[3]
    mturk_survey_file = sys.argv[4]
    mturk_recipes_file = sys.argv[5]
    gensim_model_import = sys.argv[6]
    cluster_file = sys.argv[7]
    cluster_labels_file = sys.argv[8]
    recipe_file = sys.argv[9]

    ft_model = gensim.models.FastText.load(gensim_model_import)
    ft_vocab = ft_model.wv.index2word
    ft_vectors = ft_model.wv.vectors
    print "Loaded fasttext vectors, vocab size of",str(len(ft_vocab))+"."

    parsed_survey,raw_survey = read_survey(mturk_survey_file)
    with open(mturk_survey_file.split(".csv")[0]+"_parsed.csv","w") as outf:
        parsed_survey.to_csv(outf)

    with open(surprise_file,"rb") as sof:
        ingredient_pair_surprises = scipy.sparse.load_npz(sof)
        print "Loaded ingredient surprise matrix file."

    with open(feature_freqs_file, "rb") as iffu:
        reader = unicodecsv.reader(iffu)
        feature_freqs = {}
        for row in reader:
            feature_freqs[int(row[0])] = int(float(row[1]))
        print "Loaded updated freature frequencies file."

    with open(feature_cooc_file,"rb") as cof:
        feature_coocs = scipy.sparse.load_npz(cof)
        print "Loaded feature co-occurrence matrix file."

    clustering, num_clusters, cluster_labels, cluster_name_to_index, superclusters, supercluster_names = load_clustering(cluster_file, cluster_labels_file)

    mturk_recipes = []
    with open(mturk_recipes_file,"r") as rif:
        reader = csv.reader(rif)
        for row in reader:
            mturk_recipes.append((row[0],[i.strip().replace(" ","_") for i in row[1:]]))

    do_mturk_corrs = False
    do_evaluate_dataset = True
    do_experiment1 = False
    do_experiment2 = False
    do_experiment3 = False
    do_experiment4 = False

    use_ings = False
    use_clusters = True
    use_superclusters = False
    if any("neighbourhood" in a for a in sys.argv):
        use_ings = True
        use_clusters = False
        use_superclusters = False
    surprises = {}
    mturk_recipe_ratings = []
    observed_pairs = set(zip(*ingredient_pair_surprises.nonzero()))
    for title,ings in mturk_recipes:
        mturk_recipe_ratings.append({"title":title})
        mturk_recipe_ratings[-1]["surp_ratings"] = parsed_survey["recipe"+str(len(mturk_recipe_ratings)-1)+"_surp"].tolist()
        mturk_recipe_ratings[-1]["pref_ratings"] = parsed_survey["recipe"+str(len(mturk_recipe_ratings)-1)+"_pref"].tolist()
        mturk_recipe_ratings[-1]["pref_rating_mean"] = np.sum(np.nan_to_num(np.array(mturk_recipe_ratings[-1]["pref_ratings"])))/float(len(mturk_recipe_ratings[-1][
                                                                                                                                               "pref_ratings"]))
        mturk_recipe_ratings[-1]["surp_rating_mean"] = np.sum(np.nan_to_num(np.array(mturk_recipe_ratings[-1]["surp_ratings"])))/float(len(mturk_recipe_ratings[-1][
                                                                                                                                               "surp_ratings"]))
        mturk_recipe_ratings[-1]["pref_rating_perc_high"] = np.sum(np.nan_to_num(np.array(mturk_recipe_ratings[-1]["pref_ratings"]))>=3)/float(len(mturk_recipe_ratings[-1][
                                                                                                                                               "pref_ratings"]))
        mturk_recipe_ratings[-1]["surp_rating_perc_high"] = np.sum(np.nan_to_num(np.array(mturk_recipe_ratings[-1]["surp_ratings"]))>=3)/float(len(mturk_recipe_ratings[-1]["surp_ratings"]))


        ings = append_clusters(ings, clustering, cluster_labels)
        surprise_list = []
        ings = [[i] if isinstance(i,str) else i for i in ings] #Just for this block we need all the features to be in lists, even the ones that aren't in clusters
        for (i1,i2) in itertools.combinations(ings,2):
            if i1[0] in ft_vocab and i2[0] in ft_vocab: #This is only here in the case that we're running with a model that does not contain all the ingredients in the mturk recipes, like with sample_100k
                i1_features = [(i1[0],ft_model.wv.vocab[i1[0]].index)]
                i2_features = [(i2[0],ft_model.wv.vocab[i2[0]].index)]
                #Unpack cluster and supercluster names if they're present, then calculate their surprises.
                if use_clusters and len(i1) > 1:
                    if len(i2) == 1 or not i2[1] == i1[1]: #Check to see that i1 and i2 aren't in the same cluster
                        i1_features.append((i1[1]+"_cluster",len(ft_vocab)+clustering[i1[0]]))
                if use_superclusters and len(i1) == 3:
                    if len(i2) < 3 or not i2[2] == i1[2]: #Check to see that i1 and i2 aren't in the same supercluster
                        i1_features.append((i1[2]+"_supercluster",len(ft_vocab)+num_clusters+supercluster_names.index(i1[2])))
                if use_clusters and len(i2) > 1:
                    if len(i1) == 1 or not i1[1] == i2[1]: #Check to see that i2 and i1 aren't in the same cluster
                        i2_features.append((i2[1]+"_cluster",len(ft_vocab)+clustering[i2[0]]))
                if use_superclusters and len(i2) == 3:
                    if len(i1) < 3 or not i1[2] == i2[2]: #Check to see that i1 and i2 aren't in the same supercluster
                        i2_features.append((i2[2]+"_supercluster",len(ft_vocab)+num_clusters+supercluster_names.index(i2[2])))
                if not use_ings: #We need the base features to be in the list for calculating cluster cross-membership, so we always add them and then remove here if needed.
                    i1_features = i1_features[1:] if len(i1_features) > 1 else i1_features
                    i2_features = i2_features[1:] if len(i2_features) > 1 else i2_features

                for (f1,f2) in itertools.product(i1_features,i2_features):
                    surprise_list.append(pairwise_surprise(f1[0], f2[0], f1[1], f2[1], observed_pairs, ingredient_pair_surprises, feature_freqs, feature_coocs, ft_model))


        surprises[title] = sorted(list(set(surprise_list)), key=lambda x: x[2])
        #surprises[title] = evaluate_recipe_surprise(ings, clustering, cluster_labels, ft_vocab, ft_model, num_clusters, supercluster_names, observed_pairs,
        #                                            ingredient_pair_surprises, feature_freqs, feature_coocs, use_ings=use_ings, use_clusters=use_clusters,
        #                                            use_superclusters=use_superclusters)

        pprint((title,[s for s in surprises[title]]),width=190)# if s[2] < 2]
        for perc in [100,95,90,50]:
            if len(surprises[title]):
                mturk_recipe_ratings[-1][str(perc)+"%ile_surp"] = np.percentile([s[2] for s in surprises[title]],100-perc)
            else:
                mturk_recipe_ratings[-1][str(perc) + "%ile_surp"] = float("nan")

    #Correlations within and between variables
    if do_mturk_corrs:
        cols_to_compare = ["avg_surprise_rating", "avg_cuisine_fam", "avg_cuisine_knowledge", "avg_cuisine_pref", "avg_ingredient_enjoyment"]
        #, "stdev_cuisine_fam", "stdev_cuisine_knowledge", "stdev_cuisine_pref", "stdev_ingredient_enjoyment"]
        for col_a,col_b in itertools.combinations(cols_to_compare,2):
            print col_a,"to",col_b,"correlation: {0:.2f} (p<{1:.5f})".format(*scipy.stats.pearsonr(parsed_survey[col_a], parsed_survey[col_b]))


    if do_evaluate_dataset:
        with io.open(recipe_file, mode="r", encoding="utf-8") as rf:
            recipe_json = json.loads(rf.read())
        print "Parsed recipes read in."

        evaluate_dataset_surprise(recipe_json, clustering, cluster_labels, ft_vocab, ft_model, num_clusters, supercluster_names, observed_pairs, ingredient_pair_surprises,
                                 feature_freqs, feature_coocs, use_ings=use_ings, use_clusters=use_clusters, use_superclusters=use_superclusters)

    #Recipe-level comparisons w/ h-surprise correlation
    if do_experiment1:
        experiment1(mturk_recipes,mturk_recipe_ratings)

    if do_experiment2:
        experiment2(parsed_survey,mturk_recipes,mturk_recipe_ratings, plot_scores=False)

    if do_experiment3:
        experiment3(parsed_survey,mturk_recipes,mturk_recipe_ratings, trials=1, plot=False, c_vals=(100,), use_heldout=False, hold_users=True)

    if do_experiment4:
        experiment4(parsed_survey,mturk_recipes,mturk_recipe_ratings, plot=True, hold_users=True)