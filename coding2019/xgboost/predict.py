#-*- coding:utf8 -*-
import sys
import xgboost as xgb
from dataset import get_data

if "__main__" == __name__:
   
    pred_data = sys.argv[1]
    pred_result = sys.argv[2]

    X, y = get_data("predict", pred_data)

    model = xgb.Booster(model_file="model_xgb")
    pred = model.predict(xgb.DMatrix(X))
    
    with open(pred_result, "a") as fo:
        for item in pred:
            fo.write("%.6f\n" % item)
