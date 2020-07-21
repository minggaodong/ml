#!/bin/python
#-*- coding:utf8 -*-

import tensorflow as tf
import input_data
import feature_column
import time
import sys

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("work_mod", "train", "train&predict")
tf.app.flags.DEFINE_string("train_dir", "../data/train", "train dir")
tf.app.flags.DEFINE_string("eval_dir", "../data/eval", "eval dir")
tf.app.flags.DEFINE_string("model_dir", "./model_ckpt", "model saved dir")
tf.app.flags.DEFINE_string("predict_dir", "../data/predict", "predict dir")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning_rate")
tf.app.flags.DEFINE_integer("batch_size", 50, "batch size")
tf.app.flags.DEFINE_string("model", "wnd", "train model, wide/deep/wnd")
tf.app.flags.DEFINE_string("worker_hosts", "" ,"worker_hosts split by ,")
tf.app.flags.DEFINE_string("job_name", "" ,"job name")
tf.app.flags.DEFINE_integer("task_index", 0, "task index")
tf.app.flags.DEFINE_integer("max_step", 70000, "max step")

def main(argv):
    print(time.asctime())
    base_columns,crossed_columns,deep_columns = feature_column.feature_column()
    if FLAGS.model == "wnd":
        estimator = tf.estimator.DNNLinearCombinedClassifier(
                #linear_feature_columns = crossed_columns,
                linear_feature_columns = base_columns + crossed_columns,
                linear_optimizer=tf.train.FtrlOptimizer(learning_rate=FLAGS.learning_rate,
                    l1_regularization_strength=0.0,
                    l2_regularization_strength=0.01),
                dnn_feature_columns = deep_columns,
                dnn_hidden_units = [200, 100, 100],
                dnn_optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=FLAGS.learning_rate,
                    l1_regularization_strength=0.0,
                    l2_regularization_strength=0.01),
                config=tf.estimator.RunConfig(
                    model_dir = FLAGS.model_dir,
                    save_checkpoints_steps = 500,
                    save_summary_steps = 500,
                    keep_checkpoint_max= 5,
                    log_step_count_steps= 500
                )
        )
    elif FLAGS.model == "deep":
        estimator = tf.estimator.DNNClassifier( 
                feature_columns=deep_columns,
                hidden_units=[100, 50, 50],
                optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=FLAGS.learning_rate),
                config=tf.estimator.RunConfig(
                    model_dir = FLAGS.model_dir,
                    save_checkpoints_steps = 600,
                    save_summary_steps = 600,
                    keep_checkpoint_max= 5,
                    log_step_count_steps= 600
                )
        )
    elif FLAGS.model == "wide":
        estimator = tf.estimator.LinearClassifier(
            feature_columns = base_columns + crossed_columns,
            optimizer=tf.train.FtrlOptimizer(learning_rate=FLAGS.learning_rate),
            config=tf.estimator.RunConfig(
                model_dir = FLAGS.model_dir,
                save_checkpoints_steps = 500,
                save_summary_steps = 500,
                keep_checkpoint_max= 5,
                log_step_count_steps= 500
            ))
    else:
        sys.exit(1)
    '''
    tf.train.ProximalAdagradOptimizer(
        learning_rate=0.001,
        l1_regularization_strength=0.001,
        l2_regularization_strength=0.001)
     # To apply learning rate decay, you can set dnn_optimizer to a callable:
    lambda: tf.AdamOptimizer(
        learning_rate=tf.exponential_decay(
            learning_rate=0.001,
            global_step=tf.get_global_step(),
            decay_steps=10000,
            decay_rate=0.96))
    '''
    TRAIN_FILE = FLAGS.train_dir + "/*"
    EVAL_FILE = FLAGS.eval_dir + "/*"
    PREDICT_FILE = FLAGS.predict_dir + "/predict.txt"
    num_workers = len(FLAGS.worker_hosts.split(","))
    if FLAGS.job_name == "worker":
        worker_index = FLAGS.task_index + 1
    else:
        worker_index = FLAGS.task_index
    print("job_name:%s, num_workers:%d, worker_index:%d" % (FLAGS.job_name, num_workers, worker_index))

    if FLAGS.work_mod == "train":
        train_spec = tf.estimator.TrainSpec(input_fn=lambda:input_data.train_data_input(TRAIN_FILE, FLAGS.batch_size, num_workers, worker_index),max_steps=FLAGS.max_step)
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda:input_data.eval_data_input(EVAL_FILE, FLAGS.batch_size), start_delay_secs=100, throttle_secs=150)
        result = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        print(result)
    elif FLAGS.work_mod == "predict":
        predictions = estimator.predict(input_fn=lambda:input_data.predict_data_input(PREDICT_FILE, FLAGS.batch_size))
        with open("./presult.txt", "w") as fo:
            for item in predictions:
                fo.write("%.6f\n" % (item["probabilities"][1]))

    print("end ok!")
    print(time.asctime())

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

