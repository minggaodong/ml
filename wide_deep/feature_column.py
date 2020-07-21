#!/usr/bin/env python
#-*- coding:utf8 -*-

import tensorflow as tf

# -------------------- 用于训练的特征 ------------------
# 年龄
age = tf.feature_column.numeric_column("age")
# 年龄分段
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[1018, 1025, 1030, 1035, 1040, 1045, 1050, 1055, 1060, 1065])
# 性别
gender = tf.feature_column.categorical_column_with_vocabulary_list("gender", ["400", "401", "402"])
# 二级地域
city = tf.feature_column.categorical_column_with_vocabulary_file("city", "city_area_code.txt")
# 操作系统类型
os = tf.feature_column.categorical_column_with_vocabulary_list("os", ["android", "IOS"])
# 手机品牌
brand = tf.feature_column.categorical_column_with_vocabulary_file("brand", "mobile_brand.txt")
# 登录频次
login = tf.feature_column.categorical_column_with_vocabulary_list("login",["210000","210001","210002","210003"])
# 人生状态
life = tf.feature_column.categorical_column_with_vocabulary_file("life","life_status_target.txt")
# 兴趣编码 no1
interest_category1 = tf.feature_column.categorical_column_with_vocabulary_file("interest_category1","interest_category.txt")
# 兴趣编码 no2
interest_category2 = tf.feature_column.categorical_column_with_vocabulary_file("interest_category2","interest_category.txt")
# 兴趣编码 no3
interest_category3 = tf.feature_column.categorical_column_with_vocabulary_file("interest_category3","interest_category.txt")
# 兴趣词 no1
interest_word1 = tf.feature_column.categorical_column_with_hash_bucket("interest_word1",hash_bucket_size=10000)
# 兴趣词 no2
interest_word2 = tf.feature_column.categorical_column_with_hash_bucket("interest_word2",hash_bucket_size=10000)
# 兴趣词 no3
interest_word3 = tf.feature_column.categorical_column_with_hash_bucket("interest_word3",hash_bucket_size=10000)
# 博文id
mid = tf.feature_column.categorical_column_with_hash_bucket("mid",hash_bucket_size=100000)
# 广告计划ID
adid = tf.feature_column.categorical_column_with_hash_bucket("adid",hash_bucket_size=10000)
# 访客
uid = tf.feature_column.categorical_column_with_hash_bucket("uid", hash_bucket_size=100000)
# uid = tf.feature_column.categorical_column_with_hash_bucket("uid", hash_bucket_size=1000000000)
# uid = tf.feature_column.categorical_column_with_identity(key='uid', num_buckets=10000000, default_value=0)
# 广告主
custuid = tf.feature_column.categorical_column_with_hash_bucket("custuid", hash_bucket_size=100000)
# 主信息流周请求量归一化
mainfeed_req_cnt_normal = tf.feature_column.numeric_column("mainfeed_req_cnt_normal")
# 用户周社交互动数归一化
social_cnt_normal = tf.feature_column.numeric_column("social_cnt_normal")
# 用户周导流互动数归一化
stream_cnt_normal = tf.feature_column.numeric_column("stream_cnt_normal")
# 用户周真实曝光数归一化
imp_real_cnt_normal = tf.feature_column.numeric_column("imp_real_cnt_normal")
# 用户周社交互动率(平滑)
social_ctr_bys = tf.feature_column.numeric_column("social_ctr_bys")
# 用户周导流互动率(平滑)
stream_ctr_bys = tf.feature_column.numeric_column("stream_ctr_bys")
# 广告主月社交互动次数归一化
cust_social_sum_normal = tf.feature_column.numeric_column("cust_social_sum_normal")
# 广告主月导流互动次数归一化
cust_stream_sum_normal = tf.feature_column.numeric_column("cust_stream_sum_normal")
# 广告主月真实曝光次数归一化
cust_imp_real_sum_normal = tf.feature_column.numeric_column("cust_imp_real_sum_normal")
# 广告主月社交互动率(平滑)
cust_social_ctr_bys = tf.feature_column.numeric_column("cust_social_ctr_bys")
# 广告主月导流互动率(平滑)
cust_stream_ctr_bys = tf.feature_column.numeric_column("cust_stream_ctr_bys")
# 场景
psid = tf.feature_column.categorical_column_with_vocabulary_file("psid", "psid.txt")
# 刷新方向
refresh_direct = tf.feature_column.categorical_column_with_vocabulary_list("refresh_direct", ["front", "back"])
# 用户特征向量
#user_vector = []
#for i in range(16):
#    tmp = tf.feature_column.numeric_column("user_vector_%d" % i)
#    user_vector.append(tmp)
# 用户特征向量（互动）
#user_vector_interact = []
#for i in range(128):
#    tmp = tf.feature_column.numeric_column("user_vector_interact_%d" % i)
#    user_vector_interact.append(tmp)

# -------------------- 生成各feature_column ------------------
def feature_column(typ):        
    base_columns = [age_buckets,gender,city,os,brand,login,life,interest_category1,interest_category2,interest_category3,
                    interest_word1,interest_word2,interest_word3,psid,refresh_direct]

    crossed_columns = [
        tf.feature_column.crossed_column([age_buckets,"gender"],hash_bucket_size=30),
        tf.feature_column.crossed_column([age_buckets,"gender","city"], hash_bucket_size=10000),
        tf.feature_column.crossed_column([age_buckets,"gender","life"], hash_bucket_size=1000),
        tf.feature_column.crossed_column([age_buckets,"gender","interest_category1"], hash_bucket_size=3000),
        tf.feature_column.crossed_column([age_buckets,"gender","interest_category2"], hash_bucket_size=3000),
        tf.feature_column.crossed_column([age_buckets,"gender","interest_category3"], hash_bucket_size=3000),
        tf.feature_column.crossed_column([age_buckets,"gender","interest_word1"], hash_bucket_size=10000),
        tf.feature_column.crossed_column([age_buckets,"gender","interest_word2"], hash_bucket_size=10000),
        tf.feature_column.crossed_column([age_buckets,"gender","interest_word3"], hash_bucket_size=10000),
        tf.feature_column.crossed_column([age_buckets,"gender","brand"], hash_bucket_size=600),
        tf.feature_column.crossed_column([age_buckets,"gender","psid"], hash_bucket_size=1000),
        tf.feature_column.crossed_column(["psid","refresh_direct"], hash_bucket_size=100),
        tf.feature_column.crossed_column(["psid","life"], hash_bucket_size=1000),
        tf.feature_column.crossed_column(["psid","interest_category1"], hash_bucket_size=2000),
        tf.feature_column.crossed_column(["psid","interest_category2"], hash_bucket_size=2000),
        tf.feature_column.crossed_column(["psid","interest_category3"], hash_bucket_size=2000),
        tf.feature_column.crossed_column(["psid","interest_word1"], hash_bucket_size=5000),
        tf.feature_column.crossed_column(["psid","interest_word2"], hash_bucket_size=5000),
        tf.feature_column.crossed_column(["psid","interest_word3"], hash_bucket_size=5000)
    ]

    deep_columns = [
        age,
        tf.feature_column.indicator_column(gender),
        tf.feature_column.indicator_column(city),
        tf.feature_column.indicator_column(os),
        tf.feature_column.indicator_column(brand),
        tf.feature_column.indicator_column(login),
        tf.feature_column.indicator_column(life),
        tf.feature_column.indicator_column(interest_category1),
        tf.feature_column.indicator_column(interest_category2),
        tf.feature_column.indicator_column(interest_category3),
        tf.feature_column.embedding_column(interest_word1,dimension=10),
        tf.feature_column.embedding_column(interest_word2,dimension=10),
        tf.feature_column.embedding_column(interest_word3,dimension=10),
        tf.feature_column.embedding_column(mid,dimension=20),
        tf.feature_column.embedding_column(adid,dimension=10),
        tf.feature_column.embedding_column(uid,dimension=20),
        tf.feature_column.embedding_column(custuid,dimension=20),
        mainfeed_req_cnt_normal,
        social_cnt_normal,
        stream_cnt_normal,
        imp_real_cnt_normal,
        social_ctr_bys,
        stream_ctr_bys
    ]

    if typ != "realtime":
        deep_columns += [
            cust_social_sum_normal,
            cust_stream_sum_normal,
            cust_imp_real_sum_normal
        ]
    
    deep_columns += [
        cust_social_ctr_bys,
        cust_stream_ctr_bys,
        tf.feature_column.indicator_column(psid),
        tf.feature_column.indicator_column(refresh_direct)
    ]

    # if typ == "realtime":
    #     deep_columns = [
    #         age,
    #         tf.feature_column.indicator_column(gender),
    #         tf.feature_column.indicator_column(city),
    #         tf.feature_column.indicator_column(os),
    #         tf.feature_column.indicator_column(brand),
    #         tf.feature_column.indicator_column(login),
    #         tf.feature_column.indicator_column(life),
    #         tf.feature_column.indicator_column(interest_category1),
    #         tf.feature_column.indicator_column(interest_category2),
    #         tf.feature_column.indicator_column(interest_category3),
    #         tf.feature_column.embedding_column(interest_word1,dimension=10),
    #         tf.feature_column.embedding_column(interest_word2,dimension=10),
    #         tf.feature_column.embedding_column(interest_word3,dimension=10),
    #         tf.feature_column.embedding_column(mid,dimension=20),
    #         tf.feature_column.embedding_column(adid,dimension=10),
    #         tf.feature_column.embedding_column(uid,dimension=20),
    #         tf.feature_column.embedding_column(custuid,dimension=20),
    #         mainfeed_req_cnt_normal,
    #         social_cnt_normal,
    #         stream_cnt_normal,
    #         imp_real_cnt_normal,
    #         social_ctr_bys,
    #         stream_ctr_bys,
    #         # cust_social_sum_normal,
    #         # cust_stream_sum_normal,
    #         # cust_imp_real_sum_normal,
    #         cust_social_ctr_bys,
    #         cust_stream_ctr_bys,
    #         tf.feature_column.indicator_column(psid),
    #         tf.feature_column.indicator_column(refresh_direct)
    #     ]

    #for i in range(16):
    #    deep_columns.append(user_vector[i])
    #for i in range(128):
    #    deep_columns.append(user_vector_interact[i])
        
    return base_columns,crossed_columns,deep_columns
