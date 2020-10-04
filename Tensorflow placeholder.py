'''
title : tensorflow for placeholder to show computational graph
author : Umair Riaz
course : Big Data
reference book : Applied Deep learning
date : November 6, 2019.
'''

#Create computational graph using placeholder tensor type.

import tensorflow as tf

a = tf.placeholder(tf.float32, [1])
b = tf.placeholder(tf.float32, [1])
weight1 = tf.placeholder(tf.float32, [1])
weight2 = tf.placeholder(tf.float32, [1])

# we can use constant here with placeholder but constant will never change.
'b = tf.constant(1.0)'
product1 = tf.multiply(a, weight1)
product2 = tf.multiply(b, weight2)
c = tf.add(product1, product2)
feed_dict={ a: [1], weight1: [2], b: [2], weight2: [4]}

sess = tf.Session()

# Simpliy feeding dict without using weighted graph 
'''
feed_dict = { a: [2,8], b: [2,2]}
'''

result = sess.run(c, feed_dict)
print(result)

sess.close()
