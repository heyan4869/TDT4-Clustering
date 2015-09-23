#################################################################################
#
#     __author__ = 'yanhe'
#
#     custom algorithm:
#        1. improved accuracy by using the tf-idf matrix
#        2. measure similarity by calculating euclidean distance
#
#################################################################################


import sys
import numpy as np
import scipy.sparse as sparse
from scipy import spatial
import random
import time
from operator import itemgetter


# function file_reader(): read the dev docVectors file to get term freq
# find out this function is unnecessary
def file_reader(path):
    target_file = open(path)
    return target_file


# function matrix_transfer(): transfer the docVectors to a sparse matrix
# improve the efficiency compare to function matrix_transfer_pre()
# term_freq: wordIndex1:frequency, wordIndex2:frequency, ..., wordIndexFinal:frequency
def tf_matrix_transfer():
    # TODO: change this path to sys.argv[]
    term_freq_path = 'HW2_data/HW2_dev.docVectors'
    term_freq = open(term_freq_path, 'r')
    cur_row_num = -1
    row_list = []
    col_list = []
    data_list = []
    for line in term_freq:
        cur_row_num += 1
        term_tf_pair = line.split()
        for pair in term_tf_pair:
            ele = pair.split(':')
            row_list.append(cur_row_num)
            col_list.append(int(ele[0]))
            data_list.append(int(ele[1]))
    coo_sparse_mtx = sparse.coo_matrix((data_list, (row_list, col_list)), dtype=np.float)
    print '\n' + "sparse matrix transfer finished." + '\n'
    dev_csr_mtx = coo_sparse_mtx.tocsr()
    return dev_csr_mtx


# generate the df matrix
def df_matrix_transfer():
    # TODO: change this path to sys.argv[]
    df = open('HW2_data/HW2_dev.df', 'r')
    df_list = []
    for line in df:
        term_df_pair = line.split(':')
        df_list.append(int(term_df_pair[1]))
    return df_list


# run k-means algorithm on the matrix to cluster docs
# TODO: could this function apply to different situation (done)
def k_means_docs(dev_csr_mtx, k_size):
    # dev_csr_mtx stores the tf values of documents by words
    [row_num, col_num] = dev_csr_mtx.shape

    # TODO: change the initialization of random centers (done)
    # initialize a cluster matrix with size k
    # choose k center randomly from the original matrix
    pick = random.sample(range(row_num), k_size)
    center_mtx = dev_csr_mtx[pick, :].toarray()

    num_of_round = 0
    max_sum_cos_dis = sys.maxint
    doc_nearest_dict = {}
    pre_dict_size = []
    cur_dict_size = []
    while num_of_round < 50:
        num_of_round += 1
        # use Dictionaries to store the docID to its nearest center
        doc_nearest_dict = {}

        # TODO: changed to euclidean
        # calculate the cosine similarity
        # cos_sim_mtx = cos_sim(dev_csr_mtx.toarray(), center_mtx)
        cos_sim_mtx = spatial.distance.cdist(dev_csr_mtx.toarray(), center_mtx, 'euclidean')
        cur_sum_of_cos_dis = cos_sim_mtx.min(axis=1).sum()

        max_idx_of_row = cos_sim_mtx.argmin(axis=1)
        for idx in range(0, len(max_idx_of_row)):
            if idx in pick:
                if pick.index(idx) in doc_nearest_dict:
                    doc_nearest_dict[pick.index(idx)].append(idx)
                else:
                    doc_nearest_dict[pick.index(idx)] = [idx]
            else:
                if max_idx_of_row[idx] in doc_nearest_dict:
                    doc_nearest_dict[max_idx_of_row[idx]].append(idx)
                else:
                    doc_nearest_dict[max_idx_of_row[idx]] = [idx]

        # update the k cluster center
        for idx_k in range(0, len(center_mtx)):
            center_mtx[idx_k] = dev_csr_mtx[doc_nearest_dict[idx_k], :].mean(axis=0)

        # TODO: change the converge condition
        if cur_sum_of_cos_dis < max_sum_cos_dis and max_sum_cos_dis - cur_sum_of_cos_dis > 0.5:
            # if more similar, update and continue
            max_sum_cos_dis = cur_sum_of_cos_dis
            print max_sum_cos_dis
        else:
            # if already converge, break the loop
            if cur_sum_of_cos_dis < max_sum_cos_dis and max_sum_cos_dis - cur_sum_of_cos_dis <= 0.5:
                max_sum_cos_dis = cur_sum_of_cos_dis
                print max_sum_cos_dis
                break

    # finished the k-means algorithm
    # print '\n' + "k-means for docs finished." + '\n'
    return doc_nearest_dict


# run k-means on the matrix to cluster words
# TODO: dev_csr_mtx needs to be transposed before passing to k_means_words
def k_means_words(dev_csr_mtx, k_size):
    # dev_csr_mtx stores the tf values of documents by words
    [row_num, col_num] = dev_csr_mtx.shape

    # initialize k random centers
    # choose k center randomly from the original matrix
    pick = random.sample(range(row_num), k_size)
    center_mtx = dev_csr_mtx[pick, :].toarray()

    num_of_round = 0
    max_sum_cos_dis = sys.maxint
    word_nearest_dict = {}
    pre_dict_size = []
    cur_dict_size = []
    while num_of_round < 50:
        num_of_round += 1
        # use Dictionaries to store the wordID to its nearest center
        word_nearest_dict = {}
        # print ("---------%s seconds---------" % (time.time()-start_time))

        # TODO: changed to euclidean
        # cos_sim_mtx = cos_sim(dev_csr_mtx.toarray(), center_mtx)
        cos_sim_mtx = spatial.distance.cdist(dev_csr_mtx.toarray(), center_mtx, 'euclidean')
        cur_sum_of_cos_dis = cos_sim_mtx.min(axis=1).sum()

        # find the closest center
        max_idx_of_row = cos_sim_mtx.argmin(axis=1)
        for idx in range(0, len(max_idx_of_row)):
            if idx in pick:
                if pick.index(idx) in word_nearest_dict:
                    word_nearest_dict[pick.index(idx)].append(idx)
                else:
                    word_nearest_dict[pick.index(idx)] = [idx]
            else:
                if max_idx_of_row[idx] in word_nearest_dict:
                    word_nearest_dict[max_idx_of_row[idx]].append(idx)
                else:
                    word_nearest_dict[max_idx_of_row[idx]] = [idx]

        # update the k centers
        for idx_k in range(0, len(center_mtx)):
            center_mtx[idx_k] = dev_csr_mtx[word_nearest_dict[idx_k], :].mean(axis=0)

        # check if k-means converged
        if cur_sum_of_cos_dis < max_sum_cos_dis and max_sum_cos_dis - cur_sum_of_cos_dis > 1:
            # if more similar, update and continue
            max_sum_cos_dis = cur_sum_of_cos_dis
            print max_sum_cos_dis
        else:
            # if already converge, break the loop
            if cur_sum_of_cos_dis < max_sum_cos_dis and max_sum_cos_dis - cur_sum_of_cos_dis <= 1:
                max_sum_cos_dis = cur_sum_of_cos_dis
                print max_sum_cos_dis
                break

    # finished the k-means algorithm
    # print '\n' + "k-means for words finished." + '\n'
    return word_nearest_dict


# function bipartite_cluster()
# use bipartite algorithm to generate doc and word clusters simultaneously
def bipartite_clustering():

    # step 0: get the original doc-word sparse matrix X (dev_csr_mtx)
    tf_mtx = tf_matrix_transfer()
    # start_time = time.time()
    [row_num, col_num] = tf_mtx.shape
    # get the df and idf list
    df_list = df_matrix_transfer()
    idf_mtx = np.log(np.add(1.0, np.divide(row_num, map(float, df_list))))
    # calculate the tf-idf matrix
    tf_idf_mtx = np.multiply(tf_mtx.toarray(), idf_mtx)

    # preparation for k-means word
    dev_mtx_word = sparse.csr_matrix(tf_idf_mtx).transpose()

    # initialize parameters
    num_of_round = 0
    word_k_size = 800
    doc_k_size = 200
    word_dict = {}
    doc_dict = {}
    # TODO: change to 20 rounds
    while num_of_round < 10:
        num_of_round += 1

        # step 1: k-means on columns of X and generate word cluster
        # TODO: test which k value is better
        word_dict = k_means_words(dev_mtx_word, word_k_size)

        # step 2: use word cluster on X and get X' (dev_csr_mtx_p)
        # dev_csr_mtx_p = sparse.lil_matrix((row_num, word_k_size), dtype=np.float)
        dev_mtx_p = np.zeros((row_num, word_k_size))
        for cluster_num in word_dict:
            # dev_csr_mtx_p[:, cluster_num] = dev_csr_mtx[:, word_nearest_dict.get(cluster_num)].mean(axis=1)
            dev_mtx_p[:, cluster_num] = np.asarray(tf_mtx[:, word_dict.get(cluster_num)].mean(axis=1)).reshape(row_num)

        # step 3: use k-means on rows of X' and generate doc cluster
        # TODO: test which k value is better
        doc_dict = k_means_docs(sparse.csr_matrix(dev_mtx_p), doc_k_size)

        # step 4: use doc cluster on X and get X''
        # dev_csr_mtx_pp = np.zeros((doc_k_size, col_num), dtype=np.float)
        # dev_csr_mtx_pp = sparse.lil_matrix((doc_k_size, col_num), dtype=np.float)
        dev_mtx_pp = np.zeros((doc_k_size, col_num))
        for cluster_num in doc_dict:
            # dev_csr_mtx_pp[cluster_num] = dev_csr_mtx[doc_nearest_dict.get(cluster_num), :].mean(axis=0)
            dev_mtx_pp[cluster_num, :] = np.asarray(tf_mtx[doc_dict.get(cluster_num), :].mean(axis=0)).reshape(col_num)

        # step 5: use X'' for k-means word again
        dev_mtx_word = sparse.csr_matrix(dev_mtx_pp).transpose()

    # finished bipartite clustering and get both doc clusters and word clusters
    print '\n' + "bipartite clustering finished." + '\n'

    # write the doc and cluster id into test file
    f = open('doc_cluster', 'w')
    for idx in range(0, row_num):
        for key in doc_dict:
            if idx in doc_dict.get(key):
                f.write(str(idx) + " " + str(key) + '\n')
                break

    # finished writing file process
    print '\n' + "doc cluster file writen finished." + '\n'


def cos_sim(dev_mtx, center_mtx):
    outer_prod = np.outer(np.linalg.norm(center_mtx, axis=1), np.linalg.norm(dev_mtx, axis=1))
    dot_prod = np.dot(dev_mtx, center_mtx.transpose())
    return dot_prod / outer_prod.T


# function main(): main function of the program
# TODO: notify this function
def main():
    # print '\n' + "start to reading the docVectors data." + '\n'
    bipartite_clustering()


# use this line to execute the main function
if __name__ == "__main__":
    main()


# end of the clustering process
