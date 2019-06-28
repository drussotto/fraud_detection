import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from utils import impute_nas, send_submission

train = pd.read_csv("input/dev.csv") #DEV-SAMPLE
test = pd.read_csv("input/oot0.csv")#OUT-OF-TIME SAMPLE

TARGET = 'ob_target'

model = RandomForestClassifier(random_state=20190628, n_estimators=1000)
model.fit(train.drop(TARGET, axis=1), train[TARGET])


preds = model.predict_proba(impute_nas(test))[:,1]


send_submission("rf_all_vars.csv", preds)        
