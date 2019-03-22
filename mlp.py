import numpy as np
import tensorflow as tf
import pickle
import os
os.chdir("drive")
os.chdir("tianchi")



def mlpLayer(inputdata,keep_pro):
    with tf.variable_scope("mlplayer"):
        reinput = tf.reshape(inputdata,(-1,302*302))
        w1 = tf.get_variable('w1', shape=[302*302, 300], initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        b1 = tf.get_variable('b1', shape=[1, 300], initializer=tf.zeros_initializer)
        w2 = tf.get_variable('w2', shape=[300, 8], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b2 = tf.get_variable('b2', shape=[8], initializer=tf.zeros_initializer)
        h1 = tf.nn.relu(tf.matmul(reinput,w1)+b1)
        h1_pool = tf.layers.dropout(h1,keep_pro)
        h2 = tf.matmul(h1_pool,w2)+b2
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.001)(w1))
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.001)(w2))
        return h2


def getXY(file,label,dataindex,i,j,size,istrain):
    temp = np.zeros((size,302,302))
    labelmatr = np.zeros((size,1))
    count = 0
    if istrain:
        for m in range(i,j):
            temp[count] = file[dataindex[m]].toarray().copy()
            labelmatr[count] = label[dataindex[m]]
            count += 1
        return temp,labelmatr
    else:
        for m in range(i,j):
            temp[count] = file[dataindex[m]].toarray().copy()
            count += 1
        return temp

def predict(yone,y_hat1):
    acc = tf.equal(tf.argmax(yone,1),tf.argmax(y_hat1,1))
    acc2 = tf.reduce_mean(tf.cast(acc,tf.float32))
    return acc2


def loadData(i,mold):
    with open("newlapla/laplamatrix_"+mold+str(i)+".pkl","rb") as f:
        file = pickle.load(f)
        dataindex = list(file.keys())
    return file,dataindex

def loadlabel():
    with open("label.pkl","rb") as f:
        label = pickle.load(f)
    return label

def getTestdata(i):
    with open("newlapla/laplamatrix_test{0}.pkl".format(str(i)),"rb") as f1:
        file = pickle.load(f1)
        length = len(file.keys())
        temp = np.zeros((length, 302, 302))
        count = 0
        for fileid,lapldata in file.items():
            temp[count] = lapldata.toarray().copy()
            count += 1
    return temp

epoch = 10000
lr = 1e-3
batch_size = 20
laplamatr = tf.placeholder(tf.float32,shape=[None,302,302])
x = tf.placeholder(tf.float32,shape=[302,302])
y = tf.placeholder(tf.int32,shape=[None,1])
dropout = tf.placeholder(tf.float32)
y_onehot = tf.one_hot(y,8)
y_onehot = tf.reshape(y_onehot,(-1,8))
mlplayer = mlpLayer(laplamatr,dropout)
y_hat = tf.nn.softmax(mlplayer)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=mlplayer,labels=y_onehot))
tf.add_to_collection('losses', loss)
allloss = tf.add_n(tf.get_collection('losses'))
var_1 = [var for var in tf.trainable_variables() if var.name.startswith('mlplayer')]
train_op = tf.train.AdamOptimizer(lr,beta1=0.5).minimize(allloss,var_list=[var_1])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    k = 1
    flag = False
    feature = np.eye(302)
    alllabel = loadlabel()
    data , index = loadData(k,"train")
    data_val,index_val = loadData(7,"train")
    indexlen = len(index)
    x_val,y_val = getXY(data_val,alllabel,index_val,0,1500,1500,True)
    start,end =0,20
    list1 = []
    list2 = []
    for epochnum in range(1,epoch+1):
        if flag:
            data , index= loadData(k,"train")
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
        laplx,labely = getXY(data,alllabel,index,start,end,batch_size,True)
        start += 20
        end += 20
        _,loss_ = sess.run([train_op,loss],feed_dict={laplamatr:laplx,x:feature,y:labely,dropout:0.5})
        if epochnum %200 == 0:
            acc = tf.equal(tf.argmax(y_onehot,1),tf.argmax(y_hat,1))
            acc2 = tf.reduce_mean(tf.cast(acc,tf.float32))
            printacc_train = acc2.eval({laplamatr:laplx,x:feature,y:labely,dropout:1})
            printacc_val = acc2.eval({laplamatr:x_val,x:feature,y:y_val,dropout:1})
            loss_val = sess.run(loss,feed_dict={laplamatr:x_val,x:feature,y:y_val,dropout:1})
            list1.append(loss_)
            list2.append(loss_val)
            print("epoch:{3},loss:{0:.6f},loss_val:{4:.6f},acc_t:{1:.5f},acc_val:{2:.5f}".format(loss_,printacc_train,printacc_val,epochnum,loss_val))
    with open("loss_t.pkl","wb") as f:
        pickle.dump(list1,f)
    with open("loss_val.pkl","wb") as f1:
        pickle.dump(list2,f1)
    for m in range(1, 8):
        testmatrix = getTestdata(m)
        testout = sess.run(y_hat, feed_dict={laplamatr: testmatrix, x: feature, y: y_val, dropout: 1})
        with open("newlapla/testout" + str(m) + "-" + ".pkl", "wb") as ftest:
            pickle.dump(testout, ftest)
