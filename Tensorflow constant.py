'''
title : tensorflow for constant to show computational graph
author : Umair Riaz
course : Big Data
reference book : Applied Deep learning
date : November 6, 2019.
'''

#Create computational graph using constant tensor type.

import tensorflow as tf

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])

matrix_sum  = tf.add(matrix1, matrix2)

#another example of constant similar to reference book

'''
value1 = tf.constant(2)
value2 = tf.constant(4)
total = tf.add(value1, value2)
'''

sess = tf.Session()
result = sess.run(matrix_sum)
print(result)

sess.close()
