# __author__ = 'yanhe'


import sys
import numpy as np
import scipy.sparse as sparse
from scipy import spatial


# function file_reader(): read the dev docVectors file to get term freq
# find out this function is unnecessary
# def file_reader(path):
#     target_file = open(path)
#     # term_freq = open('HW2_data/HW2_dev.docVectors')
#     # dict = open('HW2_data/HW2_dev.dict')
#     return target_file


# term_freq: wordIndex1:frequency, wordIndex2:frequency, ..., wordIndexFinal:frequency
# function matrix_transfer_pre(): transfer the docVectors to a sparse matrix
# function is relatively slow, needs to be refined
def matrix_transfer_pre():
    term_freq_path = 'HW2_data/HW2_dev.docVectors'
    dict_path = 'HW2_data/HW2_dev.dict'
    term_freq = open(term_freq_path, 'r')
    # dict = open(dict_path, 'r')
    term_freq_row_num = sum(1 for line in open(term_freq_path, 'r'))
    dict_row_num = sum(1 for line in open(dict_path, 'r'))

    # print term_freq_row_num
    # print '\n'
    # print dict_row_num

    # use uint16 data type to save memory space
    sparse_mtx = sparse.lil_matrix((term_freq_row_num, dict_row_num), dtype=np.uint16)
    cur_row_num = -1
    for line in term_freq:
        cur_row_num += 1
        # print line
        term_tf = line.split()
        # print term_tf.__len__()
        for pair in term_tf:
            ele = pair.split(':')
            # update the values in sparse_mtx
            sparse_mtx[cur_row_num, int(ele[0])] = int(ele[1])

    print '\n' + "sparse matrix transfer finished."
    print '\n'
    # transfer the generated lil_matrix to the form of csr_matrix
    dev_csr_mtx = sparse_mtx.tocsr()
    return dev_csr_mtx


# function matrix_transfer(): transfer the docVectors to a sparse matrix
# improve the efficiency compare to function matrix_transfer_pre()
# term_freq: wordIndex1:frequency, wordIndex2:frequency, ..., wordIndexFinal:frequency
def matrix_transfer():
    # TODO: change this path to sys.argv[]
    term_freq_path = 'HW2_data/HW2_dev.docVectors'
    term_freq = open(term_freq_path, 'r')
    # term_freq_row_num = sum(1 for line in open(term_freq_path, 'r'))
    cur_row_num = -1
    row_list = []
    col_list = []
    data_list = []
    max_ele = 0
    for line in term_freq:
        cur_row_num += 1
        term_tf_pair = line.split()
        for pair in term_tf_pair:
            ele = pair.split(':')
            row_list.append(cur_row_num)
            col_list.append(int(ele[0]))
            data_list.append(int(ele[1]))
            # max_ele = max(max_ele, int(ele[1]))
    coo_sparse_mtx = sparse.coo_matrix((data_list, (row_list, col_list)), dtype=np.float)
    print '\n' + "sparse matrix transfer finished."
    # print max_ele
    # print coo_sparse_mtx.shape
    dev_csr_mtx = coo_sparse_mtx.tocsr()
    return dev_csr_mtx


# run k-means algorithm on the matrix to cluster docs
# TODO: could this function apply to different situation
def k_means_docs():
    # dev_csr_mtx stores the tf values of documents by words
    dev_csr_mtx = matrix_transfer()
    [row_num, col_num] = dev_csr_mtx.shape

    # TODO: change the initialization of random centers
    # initialize a cluster_mtx with k = 50 and value range (0, 100)
    k_size = 50
    center_mtx = np.random.randint(1, 100, (k_size, col_num)).astype('float')
    num_of_round = 0
    min_sum_cos_dis = sys.maxint
    doc_nearest_dict = {}
    while num_of_round < 10:
        # use Dictionaries to store the docID to its nearest center
        doc_nearest_dict = {}
        num_of_round += 1
        # this calculates the cosine similarity
        # cos_sim_mtx = np.subtract(1.0, spatial.distance.cdist(dev_csr_mtx.toarray(), center_mtx, 'cosine'))
        # this calculates the cosine distance
        cos_sim_mtx = spatial.distance.cdist(dev_csr_mtx.toarray(), center_mtx, 'cosine')
        # print cos_sim_mtx.shape
        num_of_doc_line = -1
        cur_sum_of_cos_dis = 0
        for cos_dis_line in cos_sim_mtx:
            num_of_doc_line += 1
            num_of_cur_nearest_k = np.argmin(cos_dis_line)
            cur_sum_of_cos_dis += min(cos_dis_line)
            if num_of_cur_nearest_k in doc_nearest_dict:
                doc_nearest_dict[num_of_cur_nearest_k].append(num_of_doc_line)
            else:
                doc_nearest_dict[num_of_cur_nearest_k] = [num_of_doc_line]

        for idx_k in range(0, k_size):
            if idx_k in doc_nearest_dict:
                cur_k_doc = doc_nearest_dict.get(idx_k)
                num_of_line_in_cluster = len(cur_k_doc)
                # calculate the row wise sum and then take average
                temp_doc_sum = np.asarray(dev_csr_mtx[cur_k_doc, :].sum(0))[0].astype('float')
                center_mtx[idx_k] = np.divide(temp_doc_sum, num_of_line_in_cluster).astype('float')

        print '\n'
        if cur_sum_of_cos_dis < min_sum_cos_dis and min_sum_cos_dis - cur_sum_of_cos_dis > 10:
            # if more similar, update and continue
            min_sum_cos_dis = cur_sum_of_cos_dis
            print min_sum_cos_dis
        else:
            # if already converge, break the loop
            if cur_sum_of_cos_dis < min_sum_cos_dis and min_sum_cos_dis - cur_sum_of_cos_dis <= 10:
                min_sum_cos = cur_sum_of_cos_dis
                print min_sum_cos
                break

    # finished the k-means algorithm
    print '\n' + "k-means for docs finished." + '\n'
    return doc_nearest_dict


# run k-means on the matrix to cluster words
def k_means_words():
    # dev_csr_mtx stores the tf values of documents by words
    dev_csr_mtx = matrix_transfer().transpose()
    [row_num, col_num] = dev_csr_mtx.shape

    # initialize k random centers
    k_size = 200
    center_mtx = np.random.randint(1, 100, (k_size, col_num)).astype('float')
    num_of_round = 0
    min_sum_cos_dis = sys.maxint
    word_nearest_dict = {}
    while num_of_round < 20:
        # use Dictionaries to store the wordID to its nearest center
        word_nearest_dict = {}
        num_of_round += 1
        cos_sim_mtx = spatial.distance.cdist(dev_csr_mtx.toarray(), center_mtx, 'cosine')
        num_of_word_line = -1
        cur_sum_of_cos_dis = 0
        # find the closest center
        for cos_dis_line in cos_sim_mtx:
            num_of_word_line += 1
            num_of_cur_nearest_k = np.argmin(cos_dis_line)
            cur_sum_of_cos_dis += min(cos_dis_line)
            # add to the dict
            if num_of_cur_nearest_k in word_nearest_dict:
                word_nearest_dict[num_of_cur_nearest_k].append(num_of_word_line)
            else:
                word_nearest_dict[num_of_cur_nearest_k] = [num_of_word_line]
        # update the k centers
        for idx_k in range(0, k_size):
            if idx_k in word_nearest_dict:
                cur_k_doc = word_nearest_dict.get(idx_k)
                num_of_line_in_cluster = len(cur_k_doc)
                # calculate the row wise sum and then take average
                temp_doc_sum = np.asarray(dev_csr_mtx[cur_k_doc, :].sum(0))[0].astype('float')
                center_mtx[idx_k] = np.divide(temp_doc_sum, num_of_line_in_cluster).astype('float')

        if cur_sum_of_cos_dis < min_sum_cos_dis and min_sum_cos_dis - cur_sum_of_cos_dis > 10:
            # if more similar, update and continue
            min_sum_cos_dis = cur_sum_of_cos_dis
            print '\n'
            print min_sum_cos_dis
        else:
            # if already converge, break the loop
            if cur_sum_of_cos_dis < min_sum_cos_dis and min_sum_cos_dis - cur_sum_of_cos_dis <= 10:
                min_sum_cos = cur_sum_of_cos_dis
                print '\n'
                print min_sum_cos
                break

    # finished the k-means algorithm
    print '\n' + "k-means for words finished." + '\n'
    return word_nearest_dict


# function main(): main function of the program
# TODO: notify this function
def main():
    # term_freq = file_reader()
    # num_of_row = 0
    # for line in term_freq
    #     num_of_row += 1
    #     print line
    #     # print on a new line
    #     print '\n'
    # print num_of_row
    k_means_words()


# use this line to execute the main function
if __name__ == "__main__":
    main()