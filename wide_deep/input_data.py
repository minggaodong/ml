import tensorflow as tf
import os

feature_names = [
        "age","gender","city","os","brand","login","life",
        "interest_category1","interest_category2","interest_category3","interest_word1","interest_word2","interest_word3",
        "mid","adid",
        "uid",
        "custuid","mainfeed_req_cnt_normal",
        "social_cnt_normal","stream_cnt_normal","imp_real_cnt_normal","social_ctr_bys",
        "stream_ctr_bys","cust_social_sum_normal","cust_stream_sum_normal",
        "cust_imp_real_sum_normal","cust_social_ctr_bys","cust_stream_ctr_bys","psid","refresh_direct"
]

#for i in range(16):
#    feature_names.append("user_vector_%d" % i)
#for i in range(128):
#    feature_names.append("user_vector_interact_%d" % i)

field_defaults = [
        [0], 
        [1000], ["400"], ["30000"], ["null"], ["90199000"], ["210000"], ["null"],
        ["null"], ["null"], ["null"], ["null"], ["null"], ["null"],
        ["null"], ["null"], 
        ["null"], 
        ["null"], [0.],
        [0.], [0.], [0.], [0.002],
        [0.005], [0.], [0.],
        [0.], [0.002], [0.005],["null"],["null"]
]

#for i in range(16):
#    field_defaults.append([0.])
#for i in range(128):
#    field_defaults.append([0.])

def decode_csv(line):
    parsed_line = tf.decode_csv(line, field_defaults, use_quote_delim=False, field_delim=',')
    label = parsed_line[0]
    del parsed_line[0]
    features = parsed_line
    d = dict(zip(feature_names, features)), label
    return d

def train_data_input(data_dir, batch_size, num_workers, worker_index):
    dataset = tf.data.Dataset.list_files(data_dir,shuffle=False)
    dataset = dataset.apply(tf.contrib.data.parallel_interleave(tf.data.TextLineDataset, cycle_length=6))
    #dataset = dataset.apply(tf.data.experimental.filter_for_shard(num_workers, worker_index))
    dataset = dataset.shard(num_workers, worker_index)
    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=1000, count=None))
    dataset = dataset.apply(tf.data.experimental.map_and_batch(decode_csv, batch_size, num_parallel_calls=6))
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

def eval_data_input(data_dir, batch_size):
    dataset = tf.data.Dataset.list_files(data_dir,shuffle=False)
    dataset = dataset.apply(tf.data.experimental.parallel_interleave(tf.data.TextLineDataset, cycle_length=6))
    dataset = dataset.apply(tf.data.experimental.map_and_batch(decode_csv, batch_size, num_parallel_calls=6))
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

def serving_data_input(typ): 
    feature_map = {
        'uid': tf.placeholder(tf.string, [None, 1], 'uid'),
        # 'uid': tf.placeholder(tf.uint64, [None, 1], 'uid'),
        'custuid': tf.placeholder(tf.string, [None, 1], 'custuid'),
        'age': tf.placeholder(tf.uint32, [None, 1], 'age'),
        'gender': tf.placeholder(tf.string, [None, 1], 'gender'),
        'city': tf.placeholder(tf.string, [None, 1], 'city'),
        'mid': tf.placeholder(tf.string, [None, 1], 'mid'),
        'adid': tf.placeholder(tf.string, [None, 1], 'adid'),
        'os': tf.placeholder(tf.string, [None, 1], 'os'),
        'brand': tf.placeholder(tf.string, [None, 1], 'brand'),
        'login': tf.placeholder(tf.string, [None, 1], 'login'),
        'life': tf.placeholder(tf.string, [None, 1], 'life'),
        'interest_category1': tf.placeholder(tf.string, [None, 1], 'interest_category1'),
        'interest_category2': tf.placeholder(tf.string, [None, 1], 'interest_category2'),
        'interest_category3': tf.placeholder(tf.string, [None, 1], 'interest_category3'),
        'interest_word1': tf.placeholder(tf.string, [None, 1], 'interest_word1'),
        'interest_word2': tf.placeholder(tf.string, [None, 1], 'interest_word2'),
        'interest_word3': tf.placeholder(tf.string, [None, 1], 'interest_word3'),
        'mainfeed_req_cnt_normal': tf.placeholder(tf.float32, [None, 1], 'mainfeed_req_cnt_normal'),
        'social_cnt_normal': tf.placeholder(tf.float32, [None, 1], 'social_cnt_normal'),
        'stream_cnt_normal': tf.placeholder(tf.float32, [None, 1], 'stream_cnt_normal'),
        'imp_real_cnt_normal': tf.placeholder(tf.float32, [None, 1], 'imp_real_cnt_normal'),
        'social_ctr_bys': tf.placeholder(tf.float32, [None, 1], 'social_ctr_bys'),
        'stream_ctr_bys': tf.placeholder(tf.float32, [None, 1], 'stream_ctr_bys'),
        # 'cust_social_sum_normal': tf.placeholder(tf.float32, [None, 1], 'cust_social_sum_normal'),
        # 'cust_stream_sum_normal': tf.placeholder(tf.float32, [None, 1], 'cust_stream_sum_normal'),
        # 'cust_imp_real_sum_normal': tf.placeholder(tf.float32, [None, 1], 'cust_imp_real_sum_normal'),
        'cust_social_ctr_bys': tf.placeholder(tf.float32, [None, 1], 'cust_social_ctr_bys'),
        'cust_stream_ctr_bys': tf.placeholder(tf.float32, [None, 1], 'cust_stream_ctr_bys'),
        'psid': tf.placeholder(tf.string, [None, 1], 'psid'),
        'refresh_direct': tf.placeholder(tf.string, [None, 1], 'refresh_direct')
    }


    if typ != "realtime":
        feature_map['cust_social_sum_normal'] = tf.placeholder(tf.float32, [None, 1], 'cust_social_sum_normal')
        feature_map['cust_stream_sum_normal'] = tf.placeholder(tf.float32, [None, 1], 'cust_stream_sum_normal')
        feature_map['cust_imp_real_sum_normal'] = tf.placeholder(tf.float32, [None, 1], 'cust_imp_real_sum_normal')
    
    # if typ == "realtime":
    #     feature_map = {
    #             'uid': tf.placeholder(tf.string, [None, 1], 'uid'),
    #             'custuid': tf.placeholder(tf.string, [None, 1], 'custuid'),
    #             'age': tf.placeholder(tf.uint32, [None, 1], 'age'),
    #             'gender': tf.placeholder(tf.string, [None, 1], 'gender'),
    #             'city': tf.placeholder(tf.string, [None, 1], 'city'),
    #             'mid': tf.placeholder(tf.string, [None, 1], 'mid'),
    #             'adid': tf.placeholder(tf.string, [None, 1], 'adid'),
    #             'os': tf.placeholder(tf.string, [None, 1], 'os'),
    #             'brand': tf.placeholder(tf.string, [None, 1], 'brand'),
    #             'login': tf.placeholder(tf.string, [None, 1], 'login'),
    #             'life': tf.placeholder(tf.string, [None, 1], 'life'),
    #             'interest_category1': tf.placeholder(tf.string, [None, 1], 'interest_category1'),
    #             'interest_category2': tf.placeholder(tf.string, [None, 1], 'interest_category2'),
    #             'interest_category3': tf.placeholder(tf.string, [None, 1], 'interest_category3'),
    #             'interest_word1': tf.placeholder(tf.string, [None, 1], 'interest_word1'),
    #             'interest_word2': tf.placeholder(tf.string, [None, 1], 'interest_word2'),
    #             'interest_word3': tf.placeholder(tf.string, [None, 1], 'interest_word3'),
    #             'mainfeed_req_cnt_normal': tf.placeholder(tf.float32, [None, 1], 'mainfeed_req_cnt_normal'),
    #             'social_cnt_normal': tf.placeholder(tf.float32, [None, 1], 'social_cnt_normal'),
    #             'stream_cnt_normal': tf.placeholder(tf.float32, [None, 1], 'stream_cnt_normal'),
    #             'imp_real_cnt_normal': tf.placeholder(tf.float32, [None, 1], 'imp_real_cnt_normal'),
    #             'social_ctr_bys': tf.placeholder(tf.float32, [None, 1], 'social_ctr_bys'),
    #             'stream_ctr_bys': tf.placeholder(tf.float32, [None, 1], 'stream_ctr_bys'),
    #             # 'cust_social_sum_normal': tf.placeholder(tf.float32, [None, 1], 'cust_social_sum_normal'),
    #             # 'cust_stream_sum_normal': tf.placeholder(tf.float32, [None, 1], 'cust_stream_sum_normal'),
    #             # 'cust_imp_real_sum_normal': tf.placeholder(tf.float32, [None, 1], 'cust_imp_real_sum_normal'),
    #             'cust_social_ctr_bys': tf.placeholder(tf.float32, [None, 1], 'cust_social_ctr_bys'),
    #             'cust_stream_ctr_bys': tf.placeholder(tf.float32, [None, 1], 'cust_stream_ctr_bys'),
    #             'psid': tf.placeholder(tf.string, [None, 1], 'psid'),
    #             'refresh_direct': tf.placeholder(tf.string, [None, 1], 'refresh_direct')
    #         }

    #for i in range(16):
    #    name = "user_vector_%d" % i
    #    feature_map[name] = tf.placeholder(tf.float32, [None, 1], name)
    #for i in range(128):
    #    name = "user_vector_interact_%d" % i
    #    feature_map[name] = tf.placeholder(tf.float32, [None, 1], name)

    return feature_map