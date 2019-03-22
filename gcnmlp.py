import numpy as np
import tensorflow as tf
import pickle
import os
'''效果优于 attention '''
'''权值 feature 88%-90%左右'''
'''是否在同一个thread 中的 feature  '''
os.chdir("drive")
os.chdir("traindata225")

tf.random.set_random_seed(1)


def Convlayer(inputdata, lapal, keep_prob):
    with tf.variable_scope("layer"):
        w1 = tf.get_variable('w1', shape=[302, 128], initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        b1 = tf.get_variable('b1', shape=[1, 128], initializer=tf.zeros_initializer)
        w2 = tf.get_variable('w2', shape=[128, 128], initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        b2 = tf.get_variable('b2', shape=[1, 128], initializer=tf.zeros_initializer)
        lapal1 = tf.reshape(tf.matmul(lapal, inputdata), [tf.shape(lapal)[0] * 302, 302])
        h1 = tf.nn.relu(tf.matmul(lapal1, w1)+b1)
        h1 = tf.layers.dropout(h1, keep_prob)
        h1 = tf.reshape(h1, [tf.shape(lapal)[0], 302, 128])
        temp = tf.matmul(lapal, h1)
        temp = tf.reshape(temp, [tf.shape(temp)[0] * 302, 128])
        h2 = tf.nn.relu(tf.matmul(temp, w2)+b2)  # shape:[batch_size*302,128]
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.001)(w1))
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.001)(w2))
        return h2

def classification(inputdata, keep_prob):
    with tf.variable_scope("classi"):
        w1 = tf.get_variable('w1', shape=[302*128,200], initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        b1 = tf.get_variable('b1', shape=[200], initializer=tf.zeros_initializer)
        w2 = tf.get_variable('w2', shape=[200, 8], initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        b2 = tf.get_variable('b2', shape=[8], initializer=tf.zeros_initializer)
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.001)(w1))
        inputdata = tf.reshape(inputdata, (-1, 302, 128))
        inputdata = tf.reshape(inputdata, (-1, 302*128))
        h1 = tf.nn.relu(tf.matmul(inputdata,w1)+b1)
        h1_drop = tf.layers.dropout(h1,keep_prob)
        h2 = tf.matmul(h1_drop, w2) + b2
        return h2


def getXY(file, label, dataindex, i, j, size, istrain):
    temp = np.zeros((size, 302, 302))
    labelmatr = np.zeros((size, 1))
    count = 0
    if istrain:
        for m in range(i, j):
            temp[count] = file[dataindex[m]].toarray().copy()
            labelmatr[count] = label[dataindex[m]]
            count += 1
        return temp, labelmatr
    else:
        for m in range(i, j):
            temp[count] = file[dataindex[m]].toarray().copy()
            count += 1
        return temp


def getFeature(file, dataindex, i, j, size):
    temp = np.zeros((size, 302, 302))
    count = 0
    for m in range(i, j):
        temp[count] = file[dataindex[m]].toarray().copy()
        count += 1
    return temp


def predict(yone, y_hat1):
    acc = tf.equal(tf.argmax(yone, 1), tf.argmax(y_hat1, 1))
    acc2 = tf.reduce_mean(tf.cast(acc, tf.float32))
    return acc2


def loadData(i, mold):
    with open("newlapla/laplaadjI_" + mold + str(i) + ".pkl", "rb") as f:
        file = pickle.load(f)
        dataindex = list(file.keys())
    return file, dataindex


def loadFeature(i, mold):
    with open("newlapla/laplamatrix_" + mold + str(i) + ".pkl", "rb") as f:
        file = pickle.load(f)
    return file


def loadlabel():
    with open("label.pkl", "rb") as f:
        label = pickle.load(f)
    return label


def getTestdata(i):
    with open("laplamatrix_noneI_test" + str(i) + ".pkl", "rb") as f1:
        file = pickle.load(f1)
        length = len(file.keys())
        temp = np.zeros((length, 302, 302))
        count = 0
        for fileid, lapldata in file.items():
            temp[count] = lapldata.toarray().copy()
            count += 1
    return temp


epoch = 100000
lr = 1e-3
batch_size = 20
laplamatr = tf.placeholder(tf.float32, shape=[None, 302, 302])
x = tf.placeholder(tf.float32, shape=[None, 302, 302])
y = tf.placeholder(tf.int32, shape=[None, 1])
dropout = tf.placeholder(tf.float32)
y_onehot = tf.one_hot(y, 8)
y_onehot = tf.reshape(y_onehot, (-1, 8))
gcnlayer = Convlayer(x, laplamatr, dropout)
classificationlayer = classification(gcnlayer, dropout)
y_hat = tf.nn.softmax(classificationlayer)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=classificationlayer, labels=y_onehot))
tf.add_to_collection('losses', loss)
allloss = tf.add_n(tf.get_collection('losses'))
var_1 = [var for var in tf.trainable_variables() if var.name.startswith('layer')]
var_2 = [var for var in tf.trainable_variables() if var.name.startswith('attention')]
var_3 = [var for var in tf.trainable_variables() if var.name.startswith('classi')]
train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(allloss, var_list=[var_1 + var_2 + var_3])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    k = 1
    flag = False

    data, index = loadData(k, "train")
    data_val, index_val = loadData(7, "train")
    alllabel = loadlabel()
    indexlen = len(index)
    x_val, y_val = getXY(data_val, alllabel, index_val, 0, 1000, 1000, True)
    feature_val = getFeature(loadFeature(7, "train"), index_val, 0, 1000, 1000)
    start, end = 0, 20
    list1 = []
    list2 = []
    for epochnum in range(1, epoch + 1):
        if flag:
            data, index = loadData(k, "train")
            indexlen = len(index)
            flag = False
        if end > indexlen:
            start = 0
            end = 20
            k += 1
            if k > 6:
                k = 1
            flag = True
            continue
        laplx, labely = getXY(data, alllabel, index, start, end, batch_size, True)
        feature = getFeature(loadFeature(k, "train"), index, start, end, batch_size)
        start += 20
        end += 20
        _, loss_ = sess.run([train_op, loss], feed_dict={laplamatr: laplx, x: feature, y: labely, dropout: 0.5})
        if epochnum % 200 == 0:
            acc = tf.equal(tf.argmax(y_onehot, 1), tf.argmax(y_hat, 1))
            acc2 = tf.reduce_mean(tf.cast(acc, tf.float32))
            printacc_train = acc2.eval({laplamatr: laplx, x: feature, y: labely, dropout: 1})
            printacc_val = acc2.eval({laplamatr: x_val, x: feature_val, y: y_val, dropout: 1})
            loss_val = sess.run(loss, feed_dict={laplamatr: x_val, x: feature_val, y: y_val, dropout: 1})
            list1.append(loss_)
            list2.append(loss_val)
            print("epoch:{3},loss:{0:.6f},loss_val{4:.6f},acc_t:{1:.5f},acc_val:{2:.5f}".format(loss_, printacc_train,
                                                                                                printacc_val, epochnum,
                                                                                                loss_val))
    with open("loss_t.pkl", "wb") as f:
        pickle.dump(list1, f)
    with open("loss_val.pkl", "wb") as f1:
        pickle.dump(list2, f1)
