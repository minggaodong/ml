#-*- coding:utf8 -*-
import xgboost as xgb
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
from dataset import get_data

train_data = "/data0/vad/coding/data/csv/train.txt.csv"
eval_data = "/data0/vad/coding/data/csv/eval.txt.csv"

if "__main__" == __name__:
   
    # 训练集和测试集按9:1分割 
    X, y = get_data("train", train_data)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1, random_state=1)
    
    # 创建xgb模型
    model = xgb.XGBRegressor(
        objective = "binary:logistic",
        learning_rate = 0.08,
        n_estimators = 1000,
        max_depth = 7,
        min_child_weight = 5,
        gamma=0,
        subsample = 0.9,
        colsample_bytree = 0.7,
        seed = 27,
        n_jobs=-1,
    )
    
    eval_set = [(train_x, train_y), (test_x, test_y)]
    model.fit(train_x, train_y, eval_set=eval_set, eval_metric=["auc", "logloss"], early_stopping_rounds=20, verbose=True)
    
    # 显示特征权重
    print("-----------feature weight-------------------")
    print(model.feature_importances_)
    print("--------------------------------------------")
    
    # 保存模型
    model.save_model("./model_xgb")
