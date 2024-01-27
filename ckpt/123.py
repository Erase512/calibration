import tensorflow as tf


sess = tf.Session()


saver = tf.train.import_meta_graph('check.ckpt.meta')
saver.restore(sess, 'check.ckpt.index')


trainable_vars = tf.trainable_variables()

total_params = 0
for var in trainable_vars:
    shape = var.get_shape().as_list()
    params = 1
    for dim in shape:
        params *= dim
    total_params += params

print(f"Total Parameters: {total_params}")
