#!/usr/bin/env python
import tensorflow as tf

# @author guangbin1

#x shape:[batch_size, x0_len]
def fmLayer(x0, x0_len):

    with tf.name_scope("fm_lay") as scope:
        b = tf.Variable(tf.constant(0.1, shape = [1]), name = 'bias')
        w = tf.Variable(tf.constant(0.1, shape = [x0_len,1]), name = 'w1')
        linear_terms = tf.add(tf.matmul(x0, w), b)

        v = tf.Variable(tf.truncated_normal(shape=[x0_len, 30], mean=0, stddev=0.01),dtype='float32')
        interaction_terms = tf.multiply(0.5,
            tf.reduce_sum(
                tf.subtract(tf.pow(tf.matmul(x0, v), 2),tf.matmul(tf.pow(x0, 2), tf.pow(v, 2))),
                1, 
                keep_dims=True
            )
        )

        y_fm = tf.add(linear_terms, interaction_terms)
        return y_fm

def fmNetwork(x0, x0_len):
    with tf.name_scope("fm") as scope:
        
        fm_layer_vec = fmLayer(x0,x0_len)
        return fm_layer_vec

