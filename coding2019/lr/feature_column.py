#!/usr/bin/env python
#-*- coding:utf8 -*-

import tensorflow as tf

# -------------------- 用于训练的特征 ------------------
# 广告主特征
uid = tf.feature_column.categorical_column_with_hash_bucket("uid", hash_bucket_size=100000)
uid_gender = tf.feature_column.categorical_column_with_hash_bucket("uid_gender", hash_bucket_size=100000)
uid_loc = tf.feature_column.categorical_column_with_hash_bucket("uid_loc", hash_bucket_size=100000)
adid = tf.feature_column.categorical_column_with_hash_bucket("adid", hash_bucket_size=10000)
adid_gender = tf.feature_column.categorical_column_with_hash_bucket("adid_gender", hash_bucket_size=10000)
adid_loc = tf.feature_column.categorical_column_with_hash_bucket("adid_loc", hash_bucket_size=10000)
render = tf.feature_column.categorical_column_with_hash_bucket("render", hash_bucket_size=100)
mid = tf.feature_column.categorical_column_with_hash_bucket("mid", hash_bucket_size=10000)
scene = tf.feature_column.categorical_column_with_vocabulary_file("scene", "scene.txt")
field = tf.feature_column.categorical_column_with_vocabulary_file("field", "field.txt")
field_gender = tf.feature_column.categorical_column_with_vocabulary_file("field_gender", "field_gender.txt")

cust_freq = tf.feature_column.categorical_column_with_vocabulary_file("cust_freq", "cust_freq.txt")
mid_freq = tf.feature_column.categorical_column_with_vocabulary_list("mid_freq", ["18446744073709551615","506806140928","506806140929","506806140930","506806140931","506806140932"])

# 访客特征
gender = tf.feature_column.categorical_column_with_vocabulary_list("gender", ["1948898982394895882", "2172556925677828616", "2396214868960761350"])
loc = tf.feature_column.categorical_column_with_vocabulary_file("loc", "loc.txt")
age = tf.feature_column.categorical_column_with_vocabulary_file("age", "age.txt")

net = tf.feature_column.categorical_column_with_vocabulary_list("net", ["5292221974507643727", "6067988067470923053", "6897808927374722029","6924836312992465502","6951863694315241679"])
accountlist_size = tf.feature_column.numeric_column("accountlist_size", normalizer_fn=tf.sigmoid)
back_refresh = tf.feature_column.categorical_column_with_vocabulary_list("back_refresh", ["30786768663084","4800112145765251660","4900113840386433314"])
imp_bhv_real_num = tf.feature_column.categorical_column_with_vocabulary_list("imp_bhv_real_num",["566935683072","566935683073","566935683074","566935683075","566935683076","566935683077","566935683078","566935683079"])
login_level = tf.feature_column.categorical_column_with_vocabulary_list("login_level", ["18446744073709551615","219043332096","219043332097","219043332098"])
render_net_brand = tf.feature_column.categorical_column_with_hash_bucket("render_net_brand", hash_bucket_size=100)

lately_uid = tf.feature_column.categorical_column_with_hash_bucket("lately_uid", hash_bucket_size=100000)
lately_adid = tf.feature_column.categorical_column_with_hash_bucket("lately_adid", hash_bucket_size=10000)
lately_psid = tf.feature_column.categorical_column_with_vocabulary_file("lately_psid", "lately_psid.txt")
lately_field = tf.feature_column.categorical_column_with_vocabulary_file("lately_field", "lately_field.txt")


def feature_column():	
    base_columns = [scene,field,field_gender,cust_freq,mid_freq,login_level,gender,loc,age,net,back_refresh]
    crossed_columns = []

    deep_columns = [
        accountlist_size,
        tf.feature_column.indicator_column(scene),
        tf.feature_column.indicator_column(field),
        tf.feature_column.indicator_column(field_gender),
        tf.feature_column.indicator_column(cust_freq),
        tf.feature_column.indicator_column(mid_freq),
        tf.feature_column.indicator_column(login_level),
        tf.feature_column.indicator_column(gender),
        tf.feature_column.indicator_column(loc),
        tf.feature_column.indicator_column(age),
        tf.feature_column.indicator_column(net),
        tf.feature_column.indicator_column(back_refresh),
		
        tf.feature_column.indicator_column(imp_bhv_real_num),
        tf.feature_column.indicator_column(lately_psid),
        tf.feature_column.indicator_column(lately_field),		
		
        #tf.feature_column.embedding_column(uid, dimension=100),
        tf.feature_column.embedding_column(uid_gender, dimension=100),
        tf.feature_column.embedding_column(uid_loc, dimension=100),
        #tf.feature_column.embedding_column(adid, dimension=100),
        tf.feature_column.embedding_column(adid_gender, dimension=100),
        tf.feature_column.embedding_column(adid_loc, dimension=100),
        tf.feature_column.embedding_column(mid, dimension=100),
        tf.feature_column.embedding_column(render, dimension=50),
        tf.feature_column.embedding_column(render_net_brand, dimension=50),
        #tf.feature_column.embedding_column(lately_uid, dimension=100),
        #tf.feature_column.embedding_column(lately_adid, dimension=100),
    ]
    uid_shared_columns = tf.feature_column.shared_embedding_columns([uid, lately_uid], dimension=100)
    adid_shared_columns = tf.feature_column.shared_embedding_columns([adid, lately_adid], dimension=100)
    deep_columns += uid_shared_columns + adid_shared_columns
    
    return base_columns,crossed_columns,deep_columns
