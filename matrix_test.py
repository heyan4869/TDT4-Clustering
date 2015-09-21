__author__ = 'Yan'


import sys
import numpy as np
import random
import scipy.sparse as sparse
from scipy import spatial
from numpy import linalg as la
# from operator import itemgetter
# def column(matrix,i):
#     f = itemgetter(i)
#     return map(f,matrix)
#
# # M = [range(x,x+5) for x in range(10)]
# # assert column(M,1) == range(1,11)
#
# cur = np.array([[2, 3, 4],
#                 [4, 5, 6],
#                 [7, 8, 9]])
# nex = np.array(column(cur.transpose(), 1))
# print nex

test = np.zeros((3, 3))

row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
test_mtx = sparse.csr_matrix((data, (row, col)), shape=(3, 3))

test[:, 0] = [ 1, 2,3 ]
print test

# # test_mtx[0] = np.array([1,2,3])
# # print test_mtx[0].toarray()
#
# # print test_mtx[[0, 1], :].sum(0) * 2
# # test_mtx = test_mtx.tolil()
# test_mtx.tolil()[0, :] = np.array([2,3,4])
#
# print test_mtx.toarray()


# nearest_dict = {}
# x = [1]
# x.append(2)
# x.append(3)
# nearest_dict[0] = x
# if 0 in nearest_dict:
#     print "true"
#     nearest_dict[0].append(4)
#     print nearest_dict[0]

# center_mtx = np.random.random_integers(0, 200, (5, 5))
# # for i in range(0, 10):
# #     print center_mtx[i, :]
# center_mtx[0] = [1, 1, 1, 1, 1]
# print center_mtx

# x = []
# x.append(1)
# x.append(2)
# x.append(3)
# print min(x)
# print x.index(min(x))

# row = np.array([0, 0, 1, 2, 2, 2])
# col = np.array([0, 2, 2, 0, 1, 2])
# data = np.array([1, 2, 3, 4, 5, 6])
# test_mtx = sparse.csr_matrix((data, (row, col)), shape=(3, 3))
# # test_mtx = test_mtx.todense()
# # test_mtx = np.random.randint(0, 10, (3, 3)).astype('float')
# res_mtx = np.zeros((3, 3), dtype=np.float)
# print test_mtx.toarray()
# print '\n'
# # test_mtx[:, 0] = np.array([8, 8, 8])
# # print test_mtx
# cur = test_mtx[:, [0, 1]].sum(1)
# print cur
# print '\n'
#
# temp = np.divide(cur, 2)
# # res_mtx[:, 2] = np.divide(cur, 2)
#
# res_mtx[:, 0] = cur.reshape(3, )
# print res_mtx
# print '\n'
#
# dev_csr_mtx_p = sparse.lil_matrix((3, 3), dtype=np.float)
# print "dev"
# print dev_csr_mtx_p.toarray()




# test_mtx = np.random.randint(0, 5, (5, 6)).astype('float')
# print test_mtx.toarray()
# print test_mtx
# print '\n'

# cur = np.array([[2, 3, 4], [4, 5, 6]])
# cur = cur[0]
# print cur

# test_mtx[0] = cur
# print test_mtx
# print cur
# print '\n'
#
# cosine_one = []
# for line in test_mtx:
#     for curline in cur:
#         temp_cosine = spatial.distance.cosine(line, curline)
#         cosine_one.append(temp_cosine)
# print cosine_one
# print '\n'
#
#
# cos_sim_mtx = spatial.distance.cdist(test_mtx, cur, 'cosine')
# print cos_sim_mtx
#
#
# # norm_mtx = np.sqrt(np.linalg.norm(test_mtx, axis=0))
# # norm_mtx = np.sqrt(np.dot(test_mtx, test_mtx.transpose()).sum(1))
# norm_mtx = np.sqrt(np.square(test_mtx).sum(1))
# print norm_mtx
# print '\n'
#
# # norm_cur = np.sqrt(np.linalg.norm(cur))
# norm_cur = np.sqrt(np.dot(cur, cur.transpose()))
# print '\n'
#
# # print np.dot(norm_mtx, norm_cur)
# # print '\n'
#
# temp1 = np.dot(test_mtx, cur.transpose())
# temp2 = np.dot(norm_mtx, norm_cur)
# arr = np.divide(np.dot(test_mtx, cur.transpose()), np.dot(norm_mtx, norm_cur))
# print arr
#
# a = np.array([1,4, 2, 3])
# print np.argmin(a)


# se = np.array([1, 2, 0, 3])
# print np.where(se == 0)
# se[se == 0] = 1
# print se

# reading and writing files
# f = open('workfile', 'w')
# doc_id = 1
# c_id = 10
# f.write(str(doc_id) + " " + str(c_id) + '\n')
# f.write(str(doc_id) + " " + str(c_id))

# ran = random.sample(range(100), 10)
# print ran


# cur = np.array([[2, 2, 3], [2, 5, 6], [7, 8, 9]])
# min_of_row = cur.min(axis=1)
# print min_of_row
# # cur_sum_of_cos_dis = min_of_row.sum()
# # print cur_sum_of_cos_dis
#
# min_of_row = list(enumerate(min_of_row))
# cluster = dict(min_of_row)
# # print cluster.get(1)
# reverted_cluster = dict(zip(cluster.values(), cluster.keys()))
# print reverted_cluster.get(2)

# cur = np.array([1,2,3,4])
#
# for idx in range(0, len(cur)):
#     print cur[idx]

# row = np.array([0, 0, 1, 2, 2, 2])
# col = np.array([0, 2, 2, 0, 1, 2])
# data = np.array([1, 2, 3, 4, 5, 6])
# test_mtx = np.array(sparse.csr_matrix((data, (row, col)), shape=(3, 3)))
#
# test_mtx = np.array([[2,3,4],[5,6,7], [12, 1, 8]])
#
# cur = np.array([[5,2,7],[10, 5, 2]])
#
#
# test = 1-spatial.distance.cdist(test_mtx, cur, 'cosine')
#
# t = np.outer(np.linalg.norm(cur, axis=1),np.linalg.norm(test_mtx,axis=1))
# temp = np.dot(test_mtx, cur.transpose())
# res = temp/t.T
# print res

# a = []
# b = [2, 3, 4]
# # print len(set(a) & set(b))
# if a == []:
#     print "t"

# if pre_doc_nearest_dict == {}:
#             continue
#         else:
#             num_of_same = 0
#             for cluster_num in doc_nearest_dict:
#                 num_of_same += len(set(doc_nearest_dict[cluster_num]).intersection(pre_doc_nearest_dict[cluster_num]))


