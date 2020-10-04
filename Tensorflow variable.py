'''
title : tensorflow for variable to show computational graph
author : Umair Riaz
course : Big Data
reference book : Applied Deep learning
date : November 6, 2019.
'''

#Create computational graph using variable tensor type.

import tensorflow as tf

var_a = tf.Variable(6)
var_b = tf.Variable(8)
result = tf.add(var_a, var_b)

# First we will initialize variable. 
init = tf.global_variables_initializer()

sess = tf.Session()

# we can use this code for initialize variable. In this case, we need to initialize of each variable.
'''
sess.run(var_a.initializer)
sess.run(var_b.initializer)
'''

sess.run(init)
output = sess.run(result)
print(output)

sess.close()
