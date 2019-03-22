import numpy as np
import scipy.sparse as sp
import pickle
import pandas as pd
import itertools

def getLapla(mold):
    Degmatrix = np.eye(adjnum)
    with open("api_index.pkl","rb") as f:
        api_index = pickle.load(f)
    for i in range(1,8):
        lapladic = {}
        print(i)
        with open("amatrix_"+mold+str(i)+".pkl","rb") as f1:
            amatrix = pickle.load(f1)
        with open("tfidf_"+mold+str(i)+".pkl","rb") as f2:
            tfidf = pickle.load(f2)
            for fileid,filedata in amatrix.items():
                tempmatrix = filedata.toarray()
                filetfidf = tfidf[fileid]
                for api,idfvalue in filetfidf.items():
                    tempmatrix[0][api_index[api]] = idfvalue
                    tempmatrix[api_index[api]][0] = idfvalue
                tempdeg = np.diag(np.diag(tempmatrix))
                tempmatrix = tempmatrix - tempdeg + Degmatrix
                Deg = np.zeros((adjnum,1))
                for na in range(adjnum):
                    tp = np.count_nonzero(tempmatrix[na])
                    Deg[na] = tp
                adj = sp.coo_matrix(tempmatrix)
                d_inv_sqrt = np.power(Deg, -0.5).flatten()
                d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
                d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
                laplamatrix = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
                lapladic[fileid] = laplamatrix.copy()
            with open("laplamatrix_"+mold + str(i) + ".pkl", "wb") as f12:
                pickle.dump(lapladic, f12)



def getAdj(mold):
    for i in range(1,8):
        with open("newlapla/laplamatrix_{0}{1}.pkl".format(mold,str(i)),"rb") as f:
            file = pickle.load(f)
            lapladic = {}
            for id,matrix in file.items():
                data = matrix.data
                ones = np.ones(data.shape)
                matrix.data = ones
                Deg = matrix.sum(1)
                d_inv_sqrt = np.power(np.array(Deg), -0.5).flatten()
                d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
                d_mat_inv_sqrt = sp.diags(d_inv_sqrt,0)
                laplamatrix = d_mat_inv_sqrt.dot(matrix).dot(d_mat_inv_sqrt)
                lapladic[id] = laplamatrix.copy()
            with open("newlapla/laplaadjI_{0}{1}.pkl".format(mold,str(i)),"wb") as f1:
                pickle.dump(lapladic,f1)


def isInSameThread(mold,filenum=8):
    with open("api_index.pkl","rb") as f:
        api_index = pickle.load(f)
    for file in range(1,filenum):
        with open("filesplit_"+mold+str(file)+".pkl","rb") as f1:
            data = pickle.load(f1)
            dict_matrix = {}
            for file_id , pdfile in data.items():
                gro = pdfile.groupby("file_id")
                tempmatrix = np.zeros((adjnum,adjnum))
                setall = set()
                for _,pdfile1 in gro:
                    temp = list(pdfile1["api"].values)
                    set1 = set()
                    for api in temp:
                        set1.add(api_index[api])
                        setall.add(api_index[api])
                    apicount = list(itertools.permutations(list(set1),2))
                    for eapi in apicount:
                        tempmatrix[eapi] = 1
                for siapi in setall:
                    tempmatrix[0][siapi] = 1
                    tempmatrix[siapi][0] = 1
                dict_matrix[file_id] = sp.coo_matrix(tempmatrix).copy()
            with open("newlapla/feature_{0}{1}.pkl".format(mold,str(file)),"wb") as f2:
                pickle.dump(dict_matrix,f2)
                print(file)


def deleteOne(mold):
    for i in range(1,8):
        with open("amatrix_{0}{1}.pkl".format(mold,str(i)),"rb") as f:
            file = pickle.load(f)
            lapladic = {}
            for fileid,data in file.items():
                tempmatrix = data.toarray()
                tempmatrix = np.delete(tempmatrix,0,1)
                tempmatrix = np.delete(tempmatrix, 0, 0)
                Deg = np.zeros((301, 1))
                for na in range(301):
                    tp = np.count_nonzero(tempmatrix[na])
                    Deg[na] = tp
                adj = sp.coo_matrix(tempmatrix)
                d_inv_sqrt = np.power(Deg, -0.5).flatten()
                d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
                d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
                laplamatrix = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
                lapladic[fileid] = laplamatrix.copy()
            with open("newlapla/lapla301_{0}{1}.pkl".format(mold,str(i)), "wb") as f12:
                pickle.dump(lapladic, f12)

def deletfea301(mold):
    for i in range(1, 8):
        with open("newlapla/feature_{0}{1}.pkl".format(mold, str(i)), "rb") as f1:
            file = pickle.load(f1)
            newfea = {}
            for fileid, data in file.items():
                tempmatrix = data.toarray()
                tempmatrix = np.delete(tempmatrix, 0, 1)
                tempmatrix = np.delete(tempmatrix, 0, 0)
                adj = sp.coo_matrix(tempmatrix)
                newfea[fileid] = adj
            with open("newlapla/feature301_{0}{1}.pkl".format(mold, str(i)), "wb") as f2:
                pickle.dump(newfea, f2)
                print(i)


