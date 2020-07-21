#!/usr/bin/env python
#-*- coding:utf8 -*-

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


train_columns = ["pv","label","uid","uid_gender","uid_os","uid_brand","uid_loc","uid_net","uid_bidtype","adid","adid_gender","adid_os","adid_brand","adid_loc","adid_bidtype","bidtype","mid","render","scene","gender","loc","age","net","phone","brand","chanle","version","os_vecsion","os_source","cust_freq","mid_freq","login_level","render_net_brand","accountlist","custid_accountlist","accountlist_size","field","field_gender","field_brand","field_net","field_bid","clientid","back_refresh","imp_bhv_num","imp_bhv_real_num","lately_uid","lately_adid","lately_psid","lately_field"]

predict_columns = ["uid","uid_gender","uid_os","uid_brand","uid_loc","uid_net","uid_bidtype","adid","adid_gender","adid_os","adid_brand","adid_loc","adid_bidtype","bidtype","mid","render","scene","gender","loc","age","net","phone","brand","chanle","version","os_vecsion","os_source","cust_freq","mid_freq","login_level","render_net_brand","accountlist","custid_accountlist","accountlist_size","field","field_gender","field_brand","field_net","field_bid","clientid","back_refresh","imp_bhv_num","imp_bhv_real_num","lately_uid","lately_adid","lately_psid","lately_field"]

feature_columns = ["uid","uid_gender","uid_net","uid_bidtype","adid","adid_gender","adid_bidtype","bidtype","mid","render","scene","gender","loc","age","net","phone","brand","chanle","version","os_vecsion","os_source","cust_freq","mid_freq","login_level","render_net_brand","accountlist","custid_accountlist","accountlist_size","field","field_gender","field_brand","field_net","field_bid","clientid","back_refresh","imp_bhv_num","imp_bhv_real_num","lately_uid","lately_adid","lately_psid","lately_field"]

def get_data(data_type, data_file):
    
    data = None
    label = None
    
    # 读取数据集
    if data_type == "train":
        data = pd.read_csv(data_file, header=None, names=train_columns)
        
        # 获取lable列
        label = data['label']
        data.drop('label',axis=1,inplace=True)
        data.drop('pv',axis=1,inplace=True)

    elif data_type == "predict":
        data = pd.read_csv(data_file, header=None, names=predict_columns)
    else:
        return data, label


    # 删除不需要的特征
    data.drop('uid_loc',axis=1,inplace=True)
    data.drop('uid_os',axis=1,inplace=True)
    data.drop('uid_brand',axis=1,inplace=True)
    
    data.drop('adid_loc',axis=1,inplace=True)
    data.drop('adid_os',axis=1,inplace=True)
    data.drop('adid_brand',axis=1,inplace=True)
    
    # 特征交叉
    data['field_lately_field'] = data['field'].astype(str).values + "_" + data['lately_field'].astype(str).values
    data['age_gender'] = data['age'].astype(str).values + "_" + data['gender'].astype(str).values
    data['render_lately_psid'] = data['render'].astype(str).values + "_" + data['lately_psid'].astype(str).values
    
    # 特征编码
    labelencoder=LabelEncoder()
    data['age'] = labelencoder.fit_transform(data['age'])
    data['accountlist_size'] = labelencoder.fit_transform(data['accountlist_size'])
    data['imp_bhv_num'] = labelencoder.fit_transform(data['imp_bhv_num'])
    data['imp_bhv_real_num'] = labelencoder.fit_transform(data['imp_bhv_real_num'])
    data['field_lately_field'] = labelencoder.fit_transform(data['field_lately_field'])
    data['age_gender'] = labelencoder.fit_transform(data['age_gender'])
    data['render_lately_psid'] = labelencoder.fit_transform(data['render_lately_psid'])
    
    #for col in data.columns:
    #    data[col] = labelencoder.fit_transform(data[col])

    data = data[feature_columns].as_matrix()

    return data, label

def get_weight():
    # 权重
    weight = [0.0149536,0.01172463,0.00710936,0.01149162,0.00633175,0.00625523,0.00435311,0.11953534,0.00991216,0.05421355,0.01448061,0.00872291,0.0036475,0.01184935,0.00897254,0.00601651,0.02031682,0.00510846
,0.00446376,0.0037496,0.00576503,0.02296642,0.0361944,0.00892415,0.00441587,0.04852198,0.00611634,0.11088701,0.07892946,0.02708919,0.00653519,0.00469278,0.09294462,0.02367647,0.03133547,0.01638477,0.00469668,0.01114843,0.00657414,0.02328537,0.09570785]

    with open("./weight.tmp", "w") as fo:
        for i in range(len(feature_columns)):
            fo.write("%s:%f\n" % (feature_columns[i], weight[i]))
    os.system(" sort -t \":\" -k 2 weight.tmp -r > weight.txt")
    os.system("rm -rf weight.tmp")


if "__main__" == __name__:

    get_weight()

    X, y = get_data("train", "../data/csv/train.txt.csv_1000")
    print(X)
    print(y)

