import tensorflow as tf
from tensorflow import float32 
import os

feature_names = [
        "uid","uid_gender","uid_os","uid_brand","uid_loc","uid_net","uid_bidtype",
        "adid","adid_gender","adid_os","adid_brand","adid_loc","adid_bidtype",
        "bidtype","mid","render","scene","gender","loc","age","net","phone","brand","chanle","version","os_vecsion",
        "os_source","cust_freq","mid_freq","login_level","render_net_brand","accountlist","custid_accountlist","accountlist_size",
        "field","field_gender","field_brand","field_net","field_bid","clientid","back_refresh","imp_bhv_num","imp_bhv_real_num",
        "lately_uid","lately_adid","lately_psid","lately_field"
]
feature_num = len(feature_names)

field_defaults = [[0],[0]]
for i in range(feature_num):
    field_defaults.append(["0:0"])
field_defaults.append(["0:0"])

feature_defaults = [[""]]
for i in range(feature_num):
    feature_defaults.append([""])
feature_defaults[34] = [0.0]


def decode_csv(line):    
    parsed_line = tf.decode_csv(line, field_defaults, use_quote_delim=False, field_delim='\t')
    del parsed_line[0]
    label = parsed_line[0]
    del parsed_line[0]
   
    feature_line = ''
    for i in range(len(parsed_line)-1):
        filed = tf.string_split([parsed_line[i]], ':').values[0]
        feature_line += ',' + filed

    parsed_line = tf.decode_csv(feature_line, feature_defaults, use_quote_delim=False, field_delim=',')
    del parsed_line[0]
    d = dict(zip(feature_names, parsed_line)), label
    return d


def train_data_input(data_dir, batch_size=100, num_workers=1, worker_index=0):
    dataset = tf.data.Dataset.list_files(data_dir,shuffle=False)
    dataset = dataset.apply(tf.contrib.data.parallel_interleave(tf.data.TextLineDataset, cycle_length=6))
    dataset = dataset.shard(num_workers, worker_index)
    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=1000, count=None))
    dataset = dataset.apply(tf.data.experimental.map_and_batch(decode_csv, batch_size, num_parallel_calls=6))
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

def eval_data_input(data_dir, batch_size=100):
    dataset = tf.data.Dataset.list_files(data_dir,shuffle=False)
    dataset = dataset.apply(tf.data.experimental.parallel_interleave(tf.data.TextLineDataset, cycle_length=6))
    dataset = dataset.apply(tf.data.experimental.map_and_batch(decode_csv, batch_size, num_parallel_calls=6))
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

if __name__ == "__main__":
    features = train_data_input('data/train/*', batch_size=1)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(features))
        #print(sess.run(labels))
		
