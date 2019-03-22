import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import matplotlib.image as mpimg

a = np.random.randint(1,5,(4,4))
b = np.random.randint(1,5,(4,4))


ap = tf.placeholder(tf.float32,shape=[4,4])
bp = tf.placeholder(tf.float32,shape=[4,4])


c = tf.concat([tf.reshape(ap,(1,4,4,1)),tf.reshape(bp,(1,4,4,1))],3)
# c = tf.transpose(c,[0,2,3,1])

with tf.Session() as sess:
    cv = sess.run(c,feed_dict={ap:a,bp:b})
    print(a,b)
    print(cv)
    print(cv.shape)


