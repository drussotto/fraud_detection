import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from utils import retrieve_scores, save_hyperparameters, impute_nas, TARGET, send_submission
from time import sleep


SEED = 20190628

results = pd.read_csv("random_forest_rfe.csv")
best = results[results["grade"] == max(results["grade"])]

train = pd.read_csv("input/dev.csv").drop("id", axis=1) #DEV-SAMPLE
test = impute_nas(pd.read_csv("input/oot0.csv").drop("id", axis=1))#OUT-OF-TIME SAMPLE

X = train[best["feats"].item().split(";")]
y = train[TARGET]
test = test[best["feats"].item().split(";")]


params = {'bootstrap': [True, False],
          'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
          'max_features': ['log2', 'sqrt', None],
          'min_samples_leaf': [1, 2, 4],
          'min_samples_split': [2, 5, 10]
          }

create_file = True
for sampling in params["bootstrap"]:
    for depth in params["max_depth"]:
        for max_feat in params["max_features"]:
            for min_leaf in params["min_samples_leaf"]:
                for min_split in params["min_samples_split"]:
                    p = {'bootstrap': sampling,
                         'max_depth': depth,
                         'max_features': max_feat,
                         'min_samples_leaf': min_leaf,
                         'min_samples_split': min_split,
                         "random_state": SEED,
                         "n_estimators": 1000
                         }
                    
                    model = RandomForestClassifier(**p).fit(X, y)
                    preds = model.predict_proba(test)[:,1]
                    results = send_submission("doesnt_matter.csv", preds)
                    
                    save_hyperparameters(p, results, "grid_search_rf.csv", create_file)
                    create_file = False
                    sleep(3)
                    
                    
                    
                    
                    
                    
                    
                    
                    

