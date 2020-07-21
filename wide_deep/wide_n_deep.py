#!/bin/python
#-*- coding:utf8 -*-

import tensorflow as tf
import input_data
import feature_column
import time
import sys

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("train_dir", "./data/train", "train dir")
tf.app.flags.DEFINE_string("eval_dir", "./data/eval", "eval dir")
tf.app.flags.DEFINE_string("model_dir", "./model_ckpt", "model saved dir")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning_rate")
tf.app.flags.DEFINE_integer("batch_size", 100, "batch size")
tf.app.flags.DEFINE_string("model", "wnd", "train model, wide/deep/wnd")
tf.app.flags.DEFINE_string("worker_hosts", "" ,"worker_hosts split by ,")
tf.app.flags.DEFINE_string("job_name", "" ,"job name")
tf.app.flags.DEFINE_integer("task_index", 0, "task index")
tf.app.flags.DEFINE_string("typ", "", "是否需要去掉三个多余特征")

def main(argv):
    print(time.asctime())

    base_columns,crossed_columns,deep_columns = feature_column.feature_column(FLAGS.typ)

    if FLAGS.model == "wnd":
        estimator = tf.estimator.DNNLinearCombinedClassifier(
                #linear_feature_columns = crossed_columns,
                linear_feature_columns = base_columns + crossed_columns,
                linear_optimizer=tf.train.FtrlOptimizer(
                                 learning_rate=FLAGS.learning_rate
                #                 l2_regularization_strength=0.001
                ),
                dnn_feature_columns = deep_columns,
                #dnn_hidden_units = [200, 100, 50],
                dnn_hidden_units = [100, 50, 50],
                dnn_optimizer=tf.train.ProximalAdagradOptimizer(
                                 learning_rate=FLAGS.learning_rate
                #                 l2_regularization_strength=0.001
                ),
                #dnn_dropout=0.1,
                config=tf.estimator.RunConfig(
                    model_dir = FLAGS.model_dir,
                    save_checkpoints_steps = 5000,
                    save_summary_steps = 5000,
                    keep_checkpoint_max= 20,
                    log_step_count_steps= 5000
                )
        )
    elif FLAGS.model == "deep":
        estimator = tf.estimator.DNNClassifier( 
                feature_columns=deep_columns,
                hidden_units=[200, 100, 50],
                optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=FLAGS.learning_rate),
                config=tf.estimator.RunConfig(
                    model_dir = FLAGS.model_dir,
                    save_checkpoints_steps = 5000,
                    save_summary_steps = 5000,
                    keep_checkpoint_max= 20,
                    log_step_count_steps= 5000
                )
        )
    else:
        sys.exit(1)

    TRAIN_FILE = FLAGS.train_dir + "/*"
    EVAL_FILE = FLAGS.eval_dir + "/*"
    num_workers = len(FLAGS.worker_hosts.split(","))
    if FLAGS.job_name == "worker":
        worker_index = FLAGS.task_index + 1
    else:
        worker_index = FLAGS.task_index
    print("job_name:%s, num_workers:%d, worker_index:%d" % (FLAGS.job_name, num_workers, worker_index))

    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(input_data.serving_data_input(FLAGS.typ))
    exporter = tf.estimator.BestExporter(
          name="best_exporter",
          serving_input_receiver_fn=serving_input_receiver_fn,
          exports_to_keep=5,
          as_text=True
    )

    if FLAGS.typ == "realtime":
        train_max_steps = 2000000
    else:
        train_max_steps = 2000000

    train_spec = tf.estimator.TrainSpec(input_fn=lambda:input_data.train_data_input(TRAIN_FILE, FLAGS.batch_size, num_workers, worker_index),max_steps=train_max_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda:input_data.eval_data_input(EVAL_FILE, FLAGS.batch_size), start_delay_secs=100, throttle_secs=150, exporters=exporter)
    result = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    print(result)
    
    print("end ok!")
    print(time.asctime())

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

