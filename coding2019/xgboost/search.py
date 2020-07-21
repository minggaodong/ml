#-*- coding:utf8 -*-
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from dataset import get_data

train_data = "/data0/vad/coding/data/csv/train.txt.csv"
eval_data = "/data0/vad/coding/data/csv/eval.txt.csv"

if "__main__" == __name__:
    
    # 训练集和测试集按9:1分割 
    X, y = get_data("train", eval_data)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1, random_state=1)

    # 待测试参数
    max_depth = [5, 7, 9]
    min_child_weight = [1, 3, 5]
    gamma = [0.0, 0.1, 0.2, 0.3, 0.4]
    subsample = [0.7, 0.8, 0.9]
    colsample_bytree = [0.7, 0.8, 0.9]
    learning_rate = [0.08, 0.1, 0.2, 0.3]
    param_grid = dict(learning_rate=learning_rate)
    
    # 设置模型
    model = xgb.XGBRegressor(
        objective = "binary:logistic",
        learning_rate = 0.08,
        n_estimators = 200,
        max_depth = 7,
        min_child_weight = 5,
        gamma=0,
        subsample = 0.9,
        colsample_bytree = 0.7,
        seed = 27,
        n_jobs=-1,
    )

    # 交叉验证，找到最优参数
    gsearch = GridSearchCV(model, param_grid = param_grid, scoring='roc_auc', n_jobs=5, iid=False, cv=5)   
    gresult = gsearch.fit(train_x, train_y)
    print("Best: %f using %s" % (gresult.best_score_, gsearch.best_params_))
