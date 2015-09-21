__author__ = 'Yan'

import numpy as np
import scipy.sparse as sparse


def file_explorer():
    dev_doc_vec = open('HW2_data/HW2_dev.docVectors')
    test_doc_vec = open('HW2_data/HW2_test.docVectors')
    dev_dict = open('HW2_data/HW2_dev.dict')
    test_dict = open('HW2_data/HW2_test.dict')

    # for dev set
    dev_doc_num = sum(1 for line in open('HW2_data/HW2_dev.docVectors', 'r'))
    print "dev_doc_num is "
    print dev_doc_num
    print '\n'
    dev_uniq_word_num = sum(1 for line in open('HW2_data/HW2_dev.dict', 'r'))
    print "dev_word_num is "
    print dev_uniq_word_num
    print '\n'

    # for test set
    test_doc_num = sum(1 for line in open('HW2_data/HW2_test.docVectors', 'r'))
    print "test_doc_num is "
    print test_doc_num
    print '\n'
    test_uniq_word_num = sum(1 for line in open('HW2_data/HW2_test.dict', 'r'))
    print "test_uniq_word_num is "
    print test_uniq_word_num
    print '\n'


# function dev_matrix_transfer(): transfer the dev docVectors to a sparse matrix
def dev_matrix_transfer():
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
    dev_sparse_mtx = sparse.lil_matrix((term_freq_row_num, dict_row_num), dtype=np.uint16)
    cur_row_num = -1
    for line in term_freq:
        cur_row_num += 1
        # print line
        term_tf = line.split()
        # print term_tf.__len__()
        for pair in term_tf:
            ele = pair.split(':')
            # update the values in sparse_mtx
            dev_sparse_mtx[cur_row_num, int(ele[0])] = int(ele[1])

    print '\n' + "sparse matrix transfer finished."
    # transfer the generated lil_matrix to the form of csr_matrix
    dev_csr_mtx = dev_sparse_mtx.tocsr()
    return dev_csr_mtx


# function test_matrix_transfer(): transfer the test docVectors to a sparse matrix
def test_matrix_transfer():
    term_freq_path = 'HW2_data/HW2_test.docVectors'
    dict_path = 'HW2_data/HW2_test.dict'
    term_freq = open(term_freq_path, 'r')
    # dict = open(dict_path, 'r')
    term_freq_row_num = sum(1 for line in open(term_freq_path, 'r'))
    dict_row_num = sum(1 for line in open(dict_path, 'r'))

    # print term_freq_row_num
    # print '\n'
    # print dict_row_num

    # use uint16 data type to save memory space
    test_sparse_mtx = sparse.lil_matrix((term_freq_row_num, dict_row_num), dtype=np.uint16)
    cur_row_num = -1
    for line in term_freq:
        cur_row_num += 1
        # print line
        term_tf = line.split()
        # print term_tf.__len__()
        for pair in term_tf:
            ele = pair.split(':')
            # update the values in sparse_mtx
            test_sparse_mtx[cur_row_num, int(ele[0])] = int(ele[1])

    print '\n' + "sparse matrix transfer finished."
    # transfer the generated lil_matrix to the form of csr_matrix
    test_csr_mtx = test_sparse_mtx.tocsr()
    return test_csr_mtx


def main():
    file_explorer()

    # for dev set
    dev_csr_mtx = dev_matrix_transfer()
    dev_tfsum = dev_csr_mtx.sum()
    print "total num of words in dev"
    print dev_tfsum
    print '\n'

    nonzero_dev_idx = dev_csr_mtx.nonzero()
    print "total num of unique words in dev"
    print len(nonzero_dev_idx[0])
    print '\n'

    # for test set
    test_csr_mtx = test_matrix_transfer()
    test_tfsum = test_csr_mtx.sum()
    print "total num of words in test"
    print test_tfsum
    print '\n'

    nonzero_test_idx = test_csr_mtx.nonzero()
    print "total num of unique words in test"
    print len(nonzero_test_idx[0])
    print '\n'

    # for the first document in dev
    nonzero_first_dev = dev_csr_mtx[0].nonzero()
    print "total num of unique words in first doc of dev"
    print len(nonzero_first_dev[0])
    print '\n'

    # print the indexes of tf = 2
    first = dev_csr_mtx.getrow(0)
    for i in range(0, 14063):
        if first[0, i] == 2:
            print i


# use this line to execute the main function
if __name__ == "__main__":
    main()
