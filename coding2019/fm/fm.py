#!/bin/python
#-*- coding:utf8 -*-

import tensorflow as tf
import input_data
import feature_column
import fm_model
import os
import time
import sys

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("train_dir", "../data/train", "train dir")
tf.app.flags.DEFINE_string("eval_dir", "../data/eval", "eval dir")
tf.app.flags.DEFINE_string("model_dir", "./model_ckpt", "model saved dir")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning_rate")
tf.app.flags.DEFINE_integer("batch_size", 200, "batch size")
tf.app.flags.DEFINE_string("model", "wnd", "train model, wide/deep/wnd")
tf.app.flags.DEFINE_string("worker_hosts", "" ,"worker_hosts split by ,")
tf.app.flags.DEFINE_string("job_name", "" ,"job name")
tf.app.flags.DEFINE_integer("task_index", 0, "task index")


def main(argv):
    feature_columns = feature_column.feature_column()
    
    #model design
    estimator = tf.estimator.Estimator(
        model_fn = fm_model.model,
        params = {
            'feature_columns': feature_columns,
            'learning_rate': FLAGS.learning_rate
        },
        config = tf.estimator.RunConfig(
            model_dir = FLAGS.model_dir,
            save_checkpoints_steps = 200,
            save_summary_steps = 200,
            keep_checkpoint_max= 10,
            log_step_count_steps= 200
        )
    )

    TRAIN_FILE = FLAGS.train_dir + "/*"
    EVAL_FILE = FLAGS.eval_dir + "/*"
    num_workers = len(FLAGS.worker_hosts.split(","))
    if FLAGS.job_name == "worker":
        worker_index = FLAGS.task_index + 1
    else:
        worker_index = FLAGS.task_index
    print("job_name:%s, num_workers:%d, worker_index:%d" % (FLAGS.job_name, num_workers, worker_index))

    train_spec = tf.estimator.TrainSpec(input_fn=lambda:input_data.train_data_input(TRAIN_FILE, FLAGS.batch_size, num_workers, worker_index),max_steps=80000)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda:input_data.eval_data_input(EVAL_FILE, FLAGS.batch_size), start_delay_secs=100, throttle_secs=150)
    result = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    print(result)
    
    print("end ok!")
    print(time.asctime())

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

