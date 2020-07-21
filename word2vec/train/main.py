#coding=utf-8
from collections import defaultdict
import tensorflow as tf
import os

import model
import input_data

FLAGS = tf.app.flags.FLAGS

# run type
tf.app.flags.DEFINE_string("task_type", "train", "run type: train & infer")
tf.app.flags.DEFINE_integer("pred_top", 3, " predict top nums")

# input data
tf.app.flags.DEFINE_string("train_dir", "./train_dir", "train dir")
tf.app.flags.DEFINE_string("eval_dir", "./eval_dir", "eval dir")
tf.app.flags.DEFINE_string("infer_dir", "./infer_dir", "infer dir")

tf.app.flags.DEFINE_string("vocab_file", "./vocab.txt", "vocab file")
tf.app.flags.DEFINE_integer("batch_size", 200, "Number of batch size")
tf.app.flags.DEFINE_integer("shuffling_buffer_size", 256, "size of shuffling")
tf.app.flags.DEFINE_integer("window_size", 1000, "windows size")
tf.app.flags.DEFINE_integer("num_epochs", 1, "num of epochs for reapt")
tf.app.flags.DEFINE_integer("p_num_threads", 2, "process threads num")
tf.app.flags.DEFINE_float("sampling_rate", 2e-5, "sampling rate")

# train model
tf.app.flags.DEFINE_string("model_dir", "./model_dir", "saved model path")
tf.app.flags.DEFINE_integer("keep_ckpt_max", 10, "the maximum number of recent checkpoint file keep")
tf.app.flags.DEFINE_integer("log_steps", 1000000, "train log steps")
tf.app.flags.DEFINE_integer("max_steps", 2000000000, "train max steps")
tf.app.flags.DEFINE_integer("size", 16, "embedding size")
tf.app.flags.DEFINE_integer("neg", 5, "num neg samples")
tf.app.flags.DEFINE_float("lrate", 0.025, "learning rate")

# 分布式平台自动传入
tf.app.flags.DEFINE_string("worker_hosts", "" ,"worker_hosts split by ,")
tf.app.flags.DEFINE_string("job_name", "" ,"job name")
tf.app.flags.DEFINE_integer("task_index", 0, "task index")


def build_vocab(train_file, vocab_file):
	# 统计词出现的频次
	word_count_dict = defaultdict(int)
	with open(train_file, 'r') as train_stream:
		for line in train_stream:
			for word in line.strip().split():
				word_count_dict[word] += 1
	
	# 生成词汇文件
	words = []
	with open(vocab_file, 'w') as vocab_stream:
		for word, count in sorted(word_count_dict.items(), key=lambda x: x[1], reverse=True):
			vocab_stream.write('%s\n' % word)
			words.append(word)
	return words

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

def main(_):

	# 加载词汇
	vocabs, counts, total_count = load_vocab(FLAGS.vocab_file)
	
	# 词汇个数加1，最后一个为保留词汇，向量中表示未知词汇
	vocab_size = len(vocabs) + 1
	
	# 创建word2vec模型
	estimator = tf.estimator.Estimator(
		model_fn = model.model_fn,
		config = tf.estimator.RunConfig(
			model_dir = FLAGS.model_dir,
			keep_checkpoint_max = FLAGS.keep_ckpt_max,
			save_checkpoints_steps = FLAGS.log_steps,
			save_summary_steps = FLAGS.log_steps,
			log_step_count_steps = FLAGS.log_steps
		),
		params = {
			"embedding_size":	FLAGS.size,	
			"learning_rate": 	FLAGS.lrate,
			"num_neg_samples":	FLAGS.neg,
			"vocab_file":		FLAGS.vocab_file,
			"vocabs":		vocabs,
			"vocab_size": 		vocab_size,
			"pred_top":		FLAGS.pred_top
		})

	# 分布式平台传入参数
	num_workers = len(FLAGS.worker_hosts.split(","))
	if FLAGS.job_name == "worker":
		worker_index = FLAGS.task_index + 1
	else:
		worker_index = 0
	print("job_name:%s, num_workers:%d, worker_index:%d" % (FLAGS.job_name, num_workers, worker_index))


	# 训练和预测
	if FLAGS.task_type == 'train':
		train_spec = tf.estimator.TrainSpec(input_fn = lambda: input_data.train_data_input( 
			FLAGS.train_dir+'/*',
			batch_size=FLAGS.batch_size,
			num_epochs=FLAGS.num_epochs,
			window_size=FLAGS.window_size,
			p_num_threads=FLAGS.p_num_threads,
			num_workers=num_workers, 
			worker_index=worker_index,
			sampling_rate=FLAGS.sampling_rate,
			vocabs=vocabs,
			counts=counts,
			total_count=total_count), max_steps=FLAGS.max_steps)

		eval_spec = tf.estimator.EvalSpec(input_fn = lambda: input_data.eval_data_input(
			FLAGS.eval_dir+'/*',
			batch_size=FLAGS.batch_size,
			shuffling_buffer_size=FLAGS.shuffling_buffer_size,
			window_size=FLAGS.window_size,
			p_num_threads=FLAGS.p_num_threads), start_delay_secs=120, throttle_secs=60)

		tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
	elif FLAGS.task_type == 'infer':
		predictions = estimator.predict(input_fn = lambda: input_data.infer_data_input(
			FLAGS.infer_dir+'/*',
			batch_size=FLAGS.batch_size,
			shuffling_buffer_size=FLAGS.shuffling_buffer_size,
			p_num_threads=FLAGS.p_num_threads), predict_keys="prob")
		with open("./presult.txt", "w") as fo:
			for i, p in enumerate(predictions):
				 fo.write("%s\n" % (p['prob']))
		
	print('end ok !!!')


if __name__ == "__main__":
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run()
