import xgboost as xgb
import pandas as pd
from utils import impute_nas, send_submission

train = pd.read_csv("input/dev.csv") #DEV-SAMPLE
test = pd.read_csv("input/oot0.csv")#OUT-OF-TIME SAMPLE

TARGET = 'ob_target'

model = xgb.XGBClassifier(random_state=20190626, n_estimators=1000)
model.fit(train.drop(TARGET, axis=1), train[TARGET])


preds = model.predict_proba(impute_nas(test))[:,1]


send_submission("xgb_all_vars.csv", preds)        
