#coding=utf-8
import tensorflow as tf
import numpy as np
import os

class exporter():
	def __init__(self, 
				 embedding_size=16,
				 vocab=None,
				 model_ckpt=None,
				 output_file=None):

		self.embedding_size = embedding_size
		self.vocab = vocab
		self.model_ckpt = model_ckpt
		self.output_file = output_file

		self.build_graph()
		self.init_op()
		
		if self.model_ckpt != None and os.path.exists(self.model_ckpt):
			self.saver_head.restore(self.sess, tf.train.latest_checkpoint(self.model_ckpt))
		else:
			raise Exception, 'model_ckpt not exist!'
	
	def init_op(self):
		self.sess = tf.Session(graph=self.graph)
		self.sess.run(self.init)
		self.sess.run(self.head_table.init)

	def build_graph(self):
		self.graph = tf.Graph()
		with self.graph.as_default():
			with tf.name_scope('head'):
				self.embeddings = tf.get_variable('embeddings', shape=[len(self.vocab)+1, self.embedding_size],
					initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))

				self.head_table = tf.contrib.lookup.index_table_from_tensor(mapping=tf.convert_to_tensor(self.vocab),
																			num_oov_buckets=0,
																			default_value=len(self.vocab))

			self.init = tf.global_variables_initializer()
			self.saver_head = tf.train.Saver({'embeddings': self.embeddings})


	def output(self):
		print('--------------------------head embedding-------------------------')
		print(self.sess.run(self.embeddings))
		emb_numpy = self.embeddings.eval(session=self.sess)
		print('emb_row=%d, emb_column=%d\n' % (len(emb_numpy), len(emb_numpy[0])))
		with open(self.output_file, "w") as fo:
			for i in range(len(self.vocab)):
				fo.write('%s\t' % self.vocab[i])
				for j in range(self.embedding_size):
					if j > 0:
						fo.write(',')
					fo.write("%f" % emb_numpy[i][j])
				fo.write('\n')
