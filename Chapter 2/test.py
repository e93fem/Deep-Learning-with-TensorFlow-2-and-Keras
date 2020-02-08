import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

x = tf.placeholder("float")
y = 2 * x
data = tf.random_uniform([4, 5], minval=-1, maxval=1)
with tf.Session() as sess:
    x_data = sess.run(data)
    print(x_data)
    print(sess.run(y, feed_dict={x: x_data}))
