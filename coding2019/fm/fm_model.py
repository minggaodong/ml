#!/usr/bin/env python

# @author guangbin1


import tensorflow as tf
import fm_layer

ModelName = "fm_model"

# params: feature_columns, learning_rate
def model(features, labels, mode, params):
    fm_col = tf.feature_column.input_layer(features, params["feature_columns"])
    fm = fm_layer.fmNetwork(fm_col, fm_col.get_shape().as_list()[1])
    logits_sigmoid = tf.nn.sigmoid(fm)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
          'pred': logits_sigmoid,
        }
        export_outputs = {
          'predict': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fm,labels = labels), name='cross_entropy')
    tf.summary.scalar('loss',cross_entropy)
    
    greater = tf.greater(fm,0.5)
    output1 = tf.where(greater,fm,tf.subtract(fm,fm))

    less = tf.less(output1, 0.5)
    output2 = tf.where(less,output1,tf.div(output1,output1))

    accuracy = tf.metrics.accuracy(labels= labels, predictions = output2, name = "acc_op")
    tf.summary.scalar('accuracy',accuracy[1])

    auc = tf.metrics.auc(labels= labels, predictions = logits_sigmoid, name = "auc_op")
    tf.summary.scalar('auc',auc[1])
    metrics = {'accuracy': accuracy, 'auc': auc}

    if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=cross_entropy, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN  
    train_op = tf.train.AdagradOptimizer(params["learning_rate"]).minimize(cross_entropy, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=cross_entropy, train_op=train_op)

