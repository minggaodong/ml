import tensorflow as tf
import os
import numpy as np
from tensorflow.contrib.lookup import HashTable
from tensorflow.contrib.lookup import TextFileIdTableInitializer
from tensorflow.contrib.lookup import IdTableWithHashBuckets

def ctx_idxx(target_idx, window_size, tokens):
	ctx_range = tf.range(start=tf.maximum(tf.constant(0, dtype=tf.int32),
		target_idx-window_size),
		limit=tf.minimum(tf.size(tokens, out_type=tf.int32),
		target_idx+window_size+1),
		delta=1, dtype=tf.int32)
	
	idx = tf.case({tf.less_equal(target_idx, window_size): lambda: target_idx,
		tf.greater(target_idx, window_size): lambda: window_size},
		exclusive=True)
	
	t0 = lambda: tf.constant([], dtype=tf.int32)
	t1 = lambda: ctx_range[idx+1:]
	t2 = lambda: ctx_range[0:idx]
	t3 = lambda: tf.concat([ctx_range[0:idx], ctx_range[idx+1:]], axis=0)
	c1 = tf.logical_and(tf.equal(idx, 0),
                        tf.less(idx+1, tf.size(ctx_range, out_type=tf.int32)))
	c2 = tf.logical_and(tf.greater(idx, 0),
                        tf.equal(idx+1, tf.size(ctx_range, out_type=tf.int32)))
	c3 = tf.logical_and(tf.greater(idx, 0),
                        tf.less(idx+1, tf.size(ctx_range, out_type=tf.int32)))
	return tf.case({c1: t1, c2: t2, c3: t3}, default=t0, exclusive=True)

def concat_to_features_and_labels(tokens, window_size):
	def internal_func(features, labels, target_idx):
		ctxs = ctx_idxx(target_idx, window_size, tokens)
		label = tf.reshape(tf.gather(tokens, ctxs), [-1, 1])
		feature = tf.fill([tf.size(label)], tokens[target_idx])
		return tf.concat([features, feature], axis=0), tf.concat([labels, label], axis=0), target_idx+1
	return internal_func


def extract_examples(tokens, window_size, p_num_threads):
	
	features = tf.constant([], dtype=tf.string)
	labels = tf.constant([], shape=[0, 1], dtype=tf.string)
	target_idx = tf.constant(0, dtype=tf.int32)
	concat_func = concat_to_features_and_labels(tokens, window_size)
	max_size = tf.size(tokens, out_type=tf.int32)
	idx_below_tokens_size = lambda w, x, idx: tf.less(idx, max_size)
	
	result = tf.while_loop(
            	cond=idx_below_tokens_size,
            	body=concat_func,
            	loop_vars=[features, labels, target_idx],
            	shape_invariants=[tf.TensorShape([None]),
                              tf.TensorShape([None, 1]),
                              target_idx.get_shape()],
		parallel_iterations=p_num_threads)
	return result[0], result[1]

def sample_prob(tokens, sampling_rate, word_count_table, total_count):
	freq = word_count_table.lookup(tokens) / total_count
	return 1 - tf.sqrt(sampling_rate / freq)

def filter_tokens_mask(tokens, sampling_rate, word_count_table, total_count):
	return tf.logical_and(tf.greater(word_count_table.lookup(tokens),
									 tf.constant(0, dtype=tf.float64)),
						  tf.less(sample_prob(tokens, sampling_rate, word_count_table, total_count),
								  tf.random_uniform(shape=[tf.size(tokens)], minval=0, maxval=1, dtype=tf.float64)))

def sample_tokens(tokens, sampling_rate, word_count_table, total_count):
	return tf.boolean_mask(tokens, 
						   filter_tokens_mask(tokens, sampling_rate, word_count_table, total_count))

def decode_line(line):
	columns = tf.string_split([line], '\t')
	feature = tf.string_split(columns.values[1:], ',')
	return feature

def train_data_input(file_path, batch_size=32, num_epochs=5, window_size=1, p_num_threads=1, num_workers=1, worker_index=0, 
					 sampling_rate=0.1, vocabs=None, counts=None, total_count=0):
	
	vocab_count_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(vocabs, counts, value_dtype=tf.float64),
	           										default_value=0)
	files = tf.data.Dataset.list_files(file_path)
	dataset = (files.apply(tf.contrib.data.parallel_interleave(tf.data.TextLineDataset, cycle_length=6))
		.shard(num_workers, worker_index)
		.map(decode_line, num_parallel_calls=p_num_threads)
		.map(lambda x: sample_tokens(x.values, sampling_rate, vocab_count_table, total_count), num_parallel_calls=p_num_threads)
		.filter(lambda x: tf.size(x) > 1)
		.filter(lambda x: tf.size(x) < 400)
		.map(lambda x: extract_examples(x, window_size, p_num_threads), num_parallel_calls=p_num_threads)
		.flat_map(lambda features, labels: tf.data.Dataset.from_tensor_slices((features, labels)))
		.repeat(num_epochs)
		.batch(batch_size, drop_remainder=True))	# we need drop_remainder to statically know the batch dimension
	
	return dataset
	iterator = dataset.make_one_shot_iterator()
	return iterator.get_next()

def eval_data_input(file_path, batch_size=32, shuffling_buffer_size=10000, window_size=1, p_num_threads=1):
	
	files = tf.data.Dataset.list_files(file_path)
	dataset = (files.apply(tf.contrib.data.parallel_interleave(tf.data.TextLineDataset, cycle_length=6))
		.map(decode_line, num_parallel_calls=p_num_threads)
		.map(lambda x: extract_examples(x.values, window_size, p_num_threads), num_parallel_calls=p_num_threads)
		.flat_map(lambda features, labels: tf.data.Dataset.from_tensor_slices((features, labels)))
		.shuffle(buffer_size=shuffling_buffer_size)
		.batch(batch_size, drop_remainder=True))
	
	iterator = dataset.make_one_shot_iterator()
	return iterator.get_next()

def infer_data_input(file_path, batch_size=32, shuffling_buffer_size=10000, p_num_threads=1):
	files = tf.data.Dataset.list_files(file_path)
	dataset = (files.apply(tf.contrib.data.parallel_interleave(tf.data.TextLineDataset, cycle_length=6))
		.map(lambda x: (tf.strings.split([x]).values), num_parallel_calls=p_num_threads)
		.batch(batch_size, drop_remainder=True))

	iterator = dataset.make_one_shot_iterator()
	return iterator.get_next()


def load_vocab(vocab_file):
	vocabs = []
	counts = []
	total_count = 0

	with open(vocab_file, 'r') as vocab_stream:
		for line in vocab_stream:
			items = line.strip().split('\t', 1)
			vocabs.append(items[0])
			counts.append(int(items[1]))
			total_count += int(items[1])
	return vocabs, counts, total_count

if __name__ == "__main__":
	vocabs, counts, total_count = load_vocab('./vocab.txt')
	#feature = train_data_input('./train_dir/test*', batch_size=1,window_size=100,sampling_rate=1, vocabs=vocabs, counts=counts,total_count=total_count)
	print('-------------------------')
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		#print(sess.run(y))
		#print(sess.run(vocab_table))
		print(sess.run(feature))
		#print(sess.run(lables))
		#print(sess.run(embd))
		#print(sess.run(sim_vocab))
		print('end!!')
