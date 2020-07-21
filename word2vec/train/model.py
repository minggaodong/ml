##coding=utf-8
import tensorflow as tf
import math

def model_fn(features, labels, mode, params):
	# 加载词到id的映射
	with tf.name_scope('vocab'):
		vocab_table = tf.contrib.lookup.index_table_from_tensor(
			mapping=tf.convert_to_tensor(params['vocabs']), 
			num_oov_buckets=0,
			default_value=params['vocab_size']-1)	# 单词未定义时，默认指向词向量表的最后一个单词下标 

	# 定义隐藏层, 词汇扩展1个用来存储未知单词
	with tf.name_scope('hidden'):
		embeddings = tf.get_variable('embeddings', shape=[params['vocab_size'], params['embedding_size']],
			initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
	
	# 预测相似词
	if mode == tf.estimator.ModeKeys.PREDICT:
		# 将相似id转为词
		sparse_index_tensor = tf.string_split([tf.read_file(params['vocab_file'])], delimiter='\n')
        	index_tensor = tf.squeeze(tf.sparse_to_dense(
			sparse_index_tensor.indices,
			[1, params['vocab_size']],
			sparse_index_tensor.values,
			default_value='unknown'))
		
		# L2正则化，泛化，防止过拟合
		normalized_embeddings = tf.nn.l2_normalize(embeddings, axis=1)
		discret_features = vocab_table.lookup(features)
		valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, tf.squeeze(discret_features))

		# 用向量内积表示余弦值: 内积越大，夹角越小，余弦值越大，向量越相似
		similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
		values, preds = tf.nn.top_k(similarity, sorted=True, k=params['pred_top'])       # 计算top
		predictions={"prob": tf.gather(index_tensor, preds)}
		export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)}
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

	discret_labels = vocab_table.lookup(labels)
	discret_features = vocab_table.lookup(features)
	discret_features_embeddings = tf.nn.embedding_lookup(embeddings, discret_features)
	
	#定义输出层权重
	with tf.name_scope('weights'):
		nce_weights = tf.get_variable('nce_weights', shape=[params['vocab_size'],params['embedding_size']],
			initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(params['embedding_size'])))

	# 定义输出层偏置
	with tf.name_scope('biases'):
		nce_biases = tf.get_variable('nce_biases', shape=[params['vocab_size']],
			initializer=tf.zeros_initializer)
	
	# 定义损失函数, 采用nce
	with tf.name_scope('loss'):
		loss = tf.reduce_mean(
			tf.nn.nce_loss(weights=nce_weights,
				biases=nce_biases,
				labels=discret_labels,
				inputs=discret_features_embeddings,
				num_sampled=params['num_neg_samples'],
				num_classes=params['vocab_size']))
	
	
	# 训练，采用随机梯度下降优化
	with tf.name_scope('optimizer'):
		optimizer = (tf.train.GradientDescentOptimizer(params['learning_rate']).minimize(loss, global_step=tf.train.get_global_step()))
	
	assert mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL
	return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=optimizer)
