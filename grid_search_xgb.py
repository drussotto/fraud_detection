import pandas as pd
import xgboost as xgb
from utils import save_hyperparameters, impute_nas, TARGET, send_submission
from time import sleep


SEED = 20190628

results = pd.read_csv("xgb_rfe.csv")
best = results[results["grade"] == max(results["grade"])].iloc[0]

train = pd.read_csv("input/dev.csv").drop("id", axis=1) #DEV-SAMPLE
test = impute_nas(pd.read_csv("input/oot0.csv").drop("id", axis=1))#OUT-OF-TIME SAMPLE

X = train[best["feats"].split(";")]
y = train[TARGET]
test = test[best["feats"].split(";")]


params = {"learning_rate"    : [0.1, 0.2, 0.3] ,
          "max_depth"        : [1, 3, 4, 5, 6, 8, 10, 12, 15],
          "gamma"            : [0, 1, 5],
          "colsample_bytree" : [0.5, 0.6, 0.7, 0.8],
          "n_estimators"     : [100, 500, 1000]
          }

create_file = True
for rate in params["learning_rate"]:
    for depth in params["max_depth"]:
        for g in params["gamma"]:
            for colsample in params["colsample_bytree"]:
                for estimators in params["n_estimators"]:
                    p = {"learning_rate"    : rate,
                         "max_depth"        : depth,
                         "gamma"            : g,
                         "colsample_bytree" : colsample,
                         "n_estimators"     : estimators,
                         "random_state"     : 20180630
                         }
                    
                    model = xgb.XGBClassifier(**p).fit(X, y)
                    preds = model.predict_proba(test)[:,1]
                    results = send_submission("doesnt_matter.csv", preds)
                    
                    save_hyperparameters(p, results, "grid_search_xgb.csv", create_file)
                    create_file = False
                    sleep(5)
                    
                    
                    
                    
                    
                    
                    
                    
                    

