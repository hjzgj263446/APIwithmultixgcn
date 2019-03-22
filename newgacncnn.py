import numpy as np
import tensorflow as tf
import pickle
import os
'''两个feature通过gcn后学习到的表征concat在一起组成多通道再用cnn处理'''


os.chdir("drive")
os.chdir("traindata225")

tf.random.set_random_seed(1)


class GCN:
    def __init__(self, featurenum, adjnum, epoch, batch_size):
        self.epoch = epoch
        self.lr = 1e-3
        self.batch_size = batch_size
        self.featurenum = featurenum
        self.adjnum = adjnum
        self.weigth = {}
        self.outdim = 128

    def Convlayer(self, inputdata, lapal, keep_prob, dim, shape, act=True):
        with tf.variable_scope("layer" + str(dim)):
            for i in range(self.featurenum):
                self.weigth["weigth_layer_{0}_{1}".format(str(dim), str(i))] = self.glorot(shape)
            supports = list()
            for j in range(self.featurenum):
                pre_sup = tf.reshape(tf.matmul(lapal, inputdata[j]), [-1, shape[0]])
                pre_sup1 = tf.matmul(pre_sup, self.weigth["weigth_layer_{0}_{1}".format(str(dim), str(j))])
                supports.append(pre_sup1)
            if act:
                newinputdata = list()
                for m in range(self.featurenum):
                    h1 = tf.nn.relu(supports[m])
                    h1_drop = tf.reshape(tf.layers.dropout(h1, keep_prob), [-1, self.adjnum, shape[1]])
                    newinputdata.append(h1_drop)
                return newinputdata
            else:
                for i in range(self.featurenum):
                    supports[i] = tf.reshape(supports[i],(-1,self.adjnum,self.outdim,1))
                newinputdata = tf.concat(supports,3)
                return newinputdata

    def glorot(self, shape, name=None):
        init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
        return tf.Variable(initial, name=name)

    def zeros(self, shape, name=None):
        initial = tf.zeros(shape, dtype=tf.float32)
        return tf.Variable(initial, name=name)

    def cnnlayer(self,inputdata):
        with tf.variable_scope("cnn"):
            w = tf.get_variable('w', shape=[1,64, 2,32], dtype=tf.float32, initializer=
            tf.truncated_normal_initializer(stddev=0.01))
            b = tf.get_variable('b', shape=[32], dtype=tf.float32, initializer=
            tf.zeros_initializer)
#             w1 = tf.get_variable('w1', shape=[1, 64, 32, 32], dtype=tf.float32, initializer=
#             tf.truncated_normal_initializer(stddev=0.01))
#             b1 = tf.get_variable('b1', shape=[32], dtype=tf.float32, initializer=
#             tf.zeros_initializer)
            h = tf.nn.relu(tf.nn.conv2d(inputdata, w, strides=[1,1, 64, 1],padding="VALID") + b)
#             h_pool = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#             h1 = tf.nn.relu(tf.nn.conv2d(h_pool, w1, strides=[1, 1, 1, 1], padding="SAME") + b1)
#             h_pool1 = tf.nn.max_pool(h1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            return h

    def classification(self, inputdata, keep_prob):
        with tf.variable_scope("classi"):
            w1 = tf.get_variable('w1', shape=[9664, 300], initializer=
            tf.truncated_normal_initializer(stddev=0.1))
            b1 = tf.get_variable('b1', shape=[300], initializer=tf.zeros_initializer)
            w2 = tf.get_variable('w2', shape=[300, 8], initializer=
            tf.truncated_normal_initializer(stddev=0.1))
            b2 = tf.get_variable('b2', shape=[8], initializer=tf.zeros_initializer)
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.001)(w1))
            inputdata = tf.reshape(inputdata, (-1, 9664))
            h1 = tf.nn.relu(tf.matmul(inputdata, w1) + b1)
            h1_drop = tf.layers.dropout(h1, keep_prob)
            h2 = tf.matmul(h1_drop, w2) + b2
            return h2

    def getXY(self, file, label, dataindex, i, j, size, istrain):
        temp = np.zeros((size, self.adjnum, self.adjnum))
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

    def getFeature(self, file, dataindex, i, j, size):
        temp = np.zeros((size, self.adjnum, self.adjnum))
        count = 0
        for m in range(i, j):
            temp[count] = file[dataindex[m]].toarray().copy()
            count += 1
        return temp

    def predict(self, yone, y_hat1):
        acc = tf.equal(tf.argmax(yone, 1), tf.argmax(y_hat1, 1))
        acc2 = tf.reduce_mean(tf.cast(acc, tf.float32))
        return acc2

    def loadData(self, i, mold, filename):
        with open("newlapla/{0}_{1}{2}.pkl".format(filename, mold, i), "rb") as f:
            file = pickle.load(f)
            dataindex = list(file.keys())
        return file, dataindex

    def loadFeature(self, i, mold, filename):
        with open("newlapla/{0}_{1}{2}.pkl".format(filename, mold, i), "rb") as f:
            file = pickle.load(f)
        return file

    def loadlabel(self, ):
        with open("label.pkl", "rb") as f:
            label = pickle.load(f)
        return label

    # def getTestdata(self,i):
    #     with open("laplamatrix_noneI_test" + str(i) + ".pkl", "rb") as f1:
    #         file = pickle.load(f1)
    #         length = len(file.keys())
    #         temp = np.zeros((length, self.adjnum, self.adjnum))
    #         count = 0
    #         for fileid, lapldata in file.items():
    #             temp[count] = lapldata.toarray().copy()
    #             count += 1
    #     return temp

    def predict(self, y_hat, y_onehot):
        acc = tf.equal(tf.argmax(y_onehot, 1), tf.argmax(y_hat, 1))
        acc2 = tf.reduce_mean(tf.cast(acc, tf.float32))
        return acc2

    def train(self):

        laplamatr = tf.placeholder(tf.float32, shape=[None, self.adjnum, self.adjnum])
        x = [tf.placeholder(tf.float32, shape=[None, self.adjnum, self.adjnum], name="x" + str(i)) for i in
             range(self.featurenum)]
        y = tf.placeholder(tf.int32, shape=[None, 1])
        dropout = tf.placeholder(tf.float32)
        y_onehot = tf.one_hot(y, 8)
        y_onehot = tf.reshape(y_onehot, (-1, 8))
        gcnlayer1 = self.Convlayer(x, laplamatr, dropout, 1, [self.adjnum, self.outdim])
        gcnlayer2 = self.Convlayer(gcnlayer1, laplamatr, dropout, 2, [self.outdim, 64], False)
        cnnlayerdata = self.cnnlayer(gcnlayer2)
        classificationlayer = self.classification(cnnlayerdata, dropout)
        y_hat = tf.nn.softmax(classificationlayer)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=classificationlayer, labels=y_onehot))
        tf.add_to_collection('losses', loss)
        for key, value in self.weigth.items():
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.001)(value))
        allloss = tf.add_n(tf.get_collection('losses'))
        acc_pre = self.predict(y_hat, y_onehot)
        var_1 = [var for var in self.weigth.values()]
        var_2 = [var for var in tf.trainable_variables() if var.name.startswith('cnn')]
        var_3 = [var for var in tf.trainable_variables() if var.name.startswith('classi')]
        train_op = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(allloss, var_list=[var_1 + var_2 + var_3])

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            k = 1
            flag = False

            features_val = []
            data, index = self.loadData(k, "train", "laplaadjI")
            data_val, index_val = self.loadData(7, "train", "laplaadjI")
            alllabel = self.loadlabel()
            indexlen = len(index)
            x_val, y_val = self.getXY(data_val, alllabel, index_val, 0, 1000, 1000, True)
            feature_val1 = self.getFeature(self.loadFeature(7, "train", "laplamatrix"), index_val, 0, 1000, 1000)
            feature_val2 = self.getFeature(self.loadFeature(7, "train", "feature2-302"), index_val, 0, 1000, 1000)
            features_val.append(feature_val1)
            features_val.append(feature_val2)
            feed_dict_val = {}
            feed_dict_val[laplamatr] = x_val
            for num in range(self.featurenum):
                feed_dict_val[x[num]] = features_val[num]
            feed_dict_val[y] = y_val
            feed_dict_val[dropout] = 1

            start, end = 0, 20

            for epochnum in range(1, self.epoch + 1):
                features_train = []
                if flag:
                    data, index = self.loadData(k, "train", "laplaadjI")
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
                laplx, labely = self.getXY(data, alllabel, index, start, end, self.batch_size, True)
                feature_train1 = self.getFeature(self.loadFeature(k, "train", "laplamatrix"), index, start, end,
                                                 self.batch_size)
                feature_train2 = self.getFeature(self.loadFeature(k, "train", "feature2-302"), index, start, end,
                                                 self.batch_size)
                features_train.append(feature_train1)
                features_train.append(feature_train2)
                start += 20
                end += 20
                feed_dict = {}
                feed_dict[laplamatr] = laplx
                for num in range(self.featurenum):
                    feed_dict[x[num]] = features_train[num]
                feed_dict[y] = labely
                feed_dict[dropout] = 0.5
                _, loss_ = sess.run([train_op, loss], feed_dict=feed_dict)
                if epochnum % 200 == 0:
                    feed_dict.update({dropout: 1})
                    acc_train = sess.run(acc_pre, feed_dict=feed_dict)
                    loss_val, acc_val = sess.run([loss, acc_pre], feed_dict=feed_dict_val)
                    print(
                        "epoch:{3},loss:{0:.6f},loss_val{4:.6f},acc_t:{1:.5f},acc_val:{2:.5f}".format(loss_, acc_train,
                                                                                                      acc_val, epochnum,
                                                                                                      loss_val))


gcn1 = GCN(2, 302, 10000, 20)
gcn1.train()