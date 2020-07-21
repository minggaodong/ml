#!/usr/bin/env python
#-*- coding:utf8 -*-

import tensorflow as tf

# -------------------- 用于训练的特征 ------------------
# 广告主特征
uid = tf.feature_column.categorical_column_with_hash_bucket("uid", hash_bucket_size=100000)
uid_gender = tf.feature_column.categorical_column_with_hash_bucket("uid_gender", hash_bucket_size=100000)
uid_loc = tf.feature_column.categorical_column_with_hash_bucket("uid_loc", hash_bucket_size=100000)
uid_brand = tf.feature_column.categorical_column_with_hash_bucket("uid_brand", hash_bucket_size=100000)
uid_os = tf.feature_column.categorical_column_with_hash_bucket("uid_os", hash_bucket_size=100000)
uid_bidtype = tf.feature_column.categorical_column_with_hash_bucket("uid_bidtype", hash_bucket_size=100000)
adid = tf.feature_column.categorical_column_with_hash_bucket("adid", hash_bucket_size=10000)
adid_gender = tf.feature_column.categorical_column_with_hash_bucket("adid_gender", hash_bucket_size=10000)
adid_loc = tf.feature_column.categorical_column_with_hash_bucket("adid_loc", hash_bucket_size=10000)
adid_os = tf.feature_column.categorical_column_with_hash_bucket("adid_os", hash_bucket_size=10000)
adid_brand = tf.feature_column.categorical_column_with_hash_bucket("adid_brand", hash_bucket_size=10000)
adid_bidtype = tf.feature_column.categorical_column_with_hash_bucket("adid_bidtype", hash_bucket_size=10000)
bidtype = tf.feature_column.categorical_column_with_vocabulary_list("bidtype",["4900207300219874819","5200212384083419781"]) 
render = tf.feature_column.categorical_column_with_hash_bucket("render", hash_bucket_size=100)
mid = tf.feature_column.categorical_column_with_hash_bucket("mid", hash_bucket_size=10000)
scene = tf.feature_column.categorical_column_with_vocabulary_file("scene", "scene.txt")
field = tf.feature_column.categorical_column_with_vocabulary_file("field", "field.txt")
field_gender = tf.feature_column.categorical_column_with_vocabulary_file("field_gender", "field_gender.txt")
field_brand = tf.feature_column.categorical_column_with_vocabulary_file("field_brand","field_brand.txt")
field_net = tf.feature_column.categorical_column_with_vocabulary_file("field_net","field_net.txt")
field_bid = tf.feature_column.categorical_column_with_vocabulary_file("field_bid","field_bid.txt")
phone = tf.feature_column.categorical_column_with_vocabulary_list("phone", ["4684169303241082952","4738709505070430703","4793249711194745750","4816449970133386833","4847789917319060797","4870990176257701880","4902330119148408548","4925530378087049631","4956870325272723595","4980070584211364678","5011410531397038642","5034610790335679725","5065950737521353689","5089150992165027476","5120490939350701440","5143691198289342523","5198231404413657570","5252771606243005321","5307311812367320368","6801878792205304496"])
brand = tf.feature_column.categorical_column_with_vocabulary_list("brand",["4684169303241082952","4738709505070430703","4793249711194745750","4816449970133386833","4847789917319060797","4870990176257701880","4902330119148408548","4925530378087049631","4956870325272723595","4980070584211364678","5011410531397038642","5034610790335679725","5065950737521353689","5089150992165027476","5120490939350701440","5143691198289342523","5198231404413657570","5252771606243005321","5307311812367320368","6801878792205304496"])
cust_freq = tf.feature_column.categorical_column_with_vocabulary_file("cust_freq", "cust_freq.txt")
mid_freq = tf.feature_column.categorical_column_with_vocabulary_list("mid_freq", ["18446744073709551615","506806140928","506806140929","506806140930","506806140931","506806140932"])

# 访客特征
gender = tf.feature_column.categorical_column_with_vocabulary_list("gender", ["1948898982394895882", "2172556925677828616", "2396214868960761350"])
loc = tf.feature_column.categorical_column_with_vocabulary_file("loc", "loc.txt")
age = tf.feature_column.categorical_column_with_vocabulary_file("age", "age.txt")

net = tf.feature_column.categorical_column_with_vocabulary_list("net", ["5292221974507643727", "6067988067470923053", "6897808927374722029","6924836312992465502","6951863694315241679"])
accountlist = tf.feature_column.categorical_column_with_hash_bucket("accountlist",hash_bucket_size=10000)
custid_accountlist = tf.feature_column.categorical_column_with_hash_bucket("custid_accountlist",hash_bucket_size=100000)
accountlist_size = tf.feature_column.categorical_column_with_vocabulary_file("accountlist_size", "accountlist.txt")
back_refresh = tf.feature_column.categorical_column_with_vocabulary_list("back_refresh", ["30786768663084","4800112145765251660","4900113840386433314"])
imp_bhv_real_num = tf.feature_column.categorical_column_with_vocabulary_list("imp_bhv_real_num",["566935683072","566935683073","566935683074","566935683075","566935683076","566935683077","566935683078","566935683079"])
login_level = tf.feature_column.categorical_column_with_vocabulary_list("login_level", ["18446744073709551615","219043332096","219043332097","219043332098"])
render_net_brand = tf.feature_column.categorical_column_with_hash_bucket("render_net_brand", hash_bucket_size=100)

lately_uid = tf.feature_column.categorical_column_with_hash_bucket("lately_uid", hash_bucket_size=100000)
lately_adid = tf.feature_column.categorical_column_with_hash_bucket("lately_adid", hash_bucket_size=10000)
lately_psid = tf.feature_column.categorical_column_with_vocabulary_file("lately_psid", "lately_psid.txt")
lately_field = tf.feature_column.categorical_column_with_vocabulary_file("lately_field", "lately_field.txt")


def feature_column():	
    base_columns = [uid,uid_gender,uid_loc,adid,adid_gender,adid_loc,render,mid,scene,field,field_gender,phone,cust_freq,mid_freq,login_level,render_net_brand,gender,loc,age,net,back_refresh,lately_uid,lately_adid,lately_psid,lately_field,accountlist_size,uid_brand,uid_os,uid_bidtype,adid_brand,adid_os,adid_bidtype,bidtype,brand,accountlist,custid_accountlist,field_brand,field_net,field_bid]
    crossed_columns = [
        tf.feature_column.crossed_column(["age","gender"],hash_bucket_size=30),
        tf.feature_column.crossed_column(["scene","field"],hash_bucket_size=100),
        tf.feature_column.crossed_column(["net","field"],hash_bucket_size=100),
        tf.feature_column.crossed_column(["net","scene"],hash_bucket_size=50),
        tf.feature_column.crossed_column(["render","field"],hash_bucket_size=100),
        tf.feature_column.crossed_column(["render","scene"],hash_bucket_size=50),
        tf.feature_column.crossed_column(["render","net"],hash_bucket_size=50),
        tf.feature_column.crossed_column(["scene","back_refresh",],hash_bucket_size=100),
        tf.feature_column.crossed_column(["age","gender","loc"],hash_bucket_size=1000),
        tf.feature_column.crossed_column(["age","gender","phone"],hash_bucket_size=500),
        tf.feature_column.crossed_column(["age","gender","brand"],hash_bucket_size=500),
        tf.feature_column.crossed_column(["age","gender","login_level"],hash_bucket_size=100),
        tf.feature_column.crossed_column(["age","gender","scene"],hash_bucket_size=300),
        tf.feature_column.crossed_column(["age","gender","field"],hash_bucket_size=300),
        tf.feature_column.crossed_column(["mid","age"],hash_bucket_size=100000),
        tf.feature_column.crossed_column(["mid","gender"],hash_bucket_size=100000),
        tf.feature_column.crossed_column(["mid","loc"],hash_bucket_size=100000),
        tf.feature_column.crossed_column(["mid","net"],hash_bucket_size=100000),
        tf.feature_column.crossed_column(["mid","scene"],hash_bucket_size=100000),

    ]

    deep_columns = [
        tf.feature_column.indicator_column(accountlist_size),
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
        tf.feature_column.indicator_column(phone),
        tf.feature_column.indicator_column(back_refresh),
        tf.feature_column.indicator_column(bidtype),
        tf.feature_column.indicator_column(brand),
        tf.feature_column.indicator_column(field_bid),
        tf.feature_column.indicator_column(field_brand),
        tf.feature_column.indicator_column(field_net),
		
        tf.feature_column.indicator_column(imp_bhv_real_num),
        tf.feature_column.indicator_column(lately_psid),
        tf.feature_column.indicator_column(lately_field),		

        tf.feature_column.embedding_column(uid_brand, dimension=100),
        tf.feature_column.embedding_column(uid_os, dimension=100),
        tf.feature_column.embedding_column(uid_bidtype, dimension=100),
        tf.feature_column.embedding_column(custid_accountlist, dimension=100),

        tf.feature_column.embedding_column(adid_bidtype, dimension=100),
        tf.feature_column.embedding_column(adid_os, dimension=100),
        tf.feature_column.embedding_column(adid_brand, dimension=100),
        #tf.feature_column.embedding_column(uid, dimension=10),
        tf.feature_column.embedding_column(uid_gender, dimension=100),
        tf.feature_column.embedding_column(uid_loc, dimension=100),
        #tf.feature_column.embedding_column(adid, dimension=10),
        tf.feature_column.embedding_column(adid_gender, dimension=100),
        tf.feature_column.embedding_column(adid_loc, dimension=100),
        tf.feature_column.embedding_column(mid, dimension=100),
        tf.feature_column.embedding_column(render, dimension=5),
        tf.feature_column.embedding_column(render_net_brand, dimension=10),
        #tf.feature_column.embedding_column(lately_uid, dimension=20),
        #tf.feature_column.embedding_column(lately_adid, dimension=10),
    ]
    uid_shared_columns = tf.feature_column.shared_embedding_columns([uid, lately_uid], dimension=100)
    adid_shared_columns = tf.feature_column.shared_embedding_columns([adid, lately_adid], dimension=100)
    deep_columns += uid_shared_columns + adid_shared_columns
    
    return base_columns,crossed_columns,deep_columns
