import tensorflow as tf
import exporter

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("embedding_size", 16, "embedding size")
tf.app.flags.DEFINE_string("vocab_file", "../vocab.txt", "")
tf.app.flags.DEFINE_string("model_ckpt", "../model_dir", "model ckpt")
tf.app.flags.DEFINE_string("output_file", "../head_vector.txt", "")

def load_vocab(vocab_file):
	vocabs = []
	with open(vocab_file, 'r') as vocab_stream:
		for line in vocab_stream:
			items = line.strip().split('\t', 1)
			vocabs.append(items[0])
	return vocabs
	

def main(argv):
	imp = exporter.exporter(embedding_size=FLAGS.embedding_size,
							vocab=load_vocab(FLAGS.vocab_file),
							model_ckpt=FLAGS.model_ckpt,
							output_file=FLAGS.output_file)
	imp.output()

if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run(main)	
