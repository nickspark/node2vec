import numpy as np
import unicodecsv as csv
# import tensorflow as tf
import math
import os, sys
import random
from scipy.sparse import csr_matrix
from tqdm import tqdm
import networkx as nx
import json
import itertools
import pandas as pd
import time
import multiprocessing as mp
from gensim.models import Word2Vec

data_path = './dataset/'


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


def create_sparse_mat(dataset):
    try:
        co_author_matrix = load_sparse_csr(data_path + dataset + '.npz')
        return co_author_matrix
    except:
        paper_author = pd.read_csv(data_path + dataset + '/paper_author.txt', header=None, sep=' ')
        paper_author.columns = ['paper_id', 'author_id']
        # def return_sparse_mat(dataset, dic1, dic2):
        author_paper_indices = []
        author_paper_values = []
        # author_paper_shape=(max(paper_author['author_id'])+1, max(paper_author['paper_id'])+1)
        author_paper_shape = (max(paper_author['author_id']) + max(paper_author['paper_id']) + 1,
                              max(paper_author['author_id']) + max(paper_author['paper_id']) + 1)
        # print(author_paper_shape)
        for i in paper_author.index:
            author_paper_indices.append([paper_author.loc[i, 'author_id'], paper_author.loc[i, 'paper_id']])
            author_paper_values.append(1)
        indeces = np.array(author_paper_indices).T
        # print(max(indeces[0]))
        # print(max(indeces[1]))
        author_paper = csr_matrix((author_paper_values, indeces), shape=author_paper_shape, dtype=np.int8)
        co_author_matrix = np.dot(author_paper, author_paper.T)
        for i in range(co_author_matrix.shape[0]):
            co_author_matrix[i, i] = 0
        co_author_matrix.eliminate_zeros()
        np.savez(data_path + dataset + '.npz', data=co_author_matrix.data, indices=co_author_matrix.indices,
                 indptr=co_author_matrix.indptr, shape=co_author_matrix.shape)
        return co_author_matrix
    return co_author_matrix


def get_random_walk(p):
    random_walks = []
    # get random walks
    for u in tqdm(range(num_nodes)):
        if len(indices[indptr[u]:indptr[u + 1]]) != 0:
            possible_next_node = indices[indptr[u]:indptr[u + 1]]
            weight_for_next_move = data[indptr[u]:indptr[u + 1]]  # i.e  possible next ndoes from u
            weight_for_next_move[weight_for_next_move < 0] = 0.0000001
            weight_for_next_move[np.isinf(weight_for_next_move)] = 0.0000001
            weight_for_next_move[np.isnan(weight_for_next_move)] = 0.0000001
            weight_for_next_move = weight_for_next_move.astype(np.float32) / np.sum(weight_for_next_move)
            weight_for_next_move[np.isnan(weight_for_next_move)] = 1
            try:
                first_walk = np.random.choice(possible_next_node, 1, p=weight_for_next_move)
            except:
                print(possible_next_node, weight_for_next_move)

            random_walk = [u, first_walk[0]]
            for i in range(random_walk_length - 2):
                cur_node = random_walk[-1]
                precious_node = random_walk[-2]
                (pi_vx_indices, pi_vx_values) = transition[precious_node, cur_node]
                # print(pi_vx_values)

                next_node = np.random.choice(pi_vx_indices, 1, p=pi_vx_values)
                random_walk.append(next_node[0])
            random_walks.append(random_walk)

    return random_walks


class MySentences(object):
    def __init__(self, np_random_walks):
        self.np_random_walks = np_random_walks

    def __iter__(self):
        for i in range(self.np_random_walks.shape[0]):
            tmp = ''
            if i % 100000 == 0:
                print('iter ' + str(i))
            yield list(map(str, np_random_walks[i]))


def alpha(p, q, t, x):
    if t == x:
        return 1.0 / p
    elif adj_mat_csr_sparse[t, x] > 0:
        return 1.0
    else:
        return 1.0 / q


p = 1.0
q = 0.5
transition = {}
# dataset = 'net_aminer_part'
dataset = 'net_aminer_homo'
# dataset = 'net_dbis_new'
co_author_matrix = create_sparse_mat(dataset)
adj_mat_csr_sparse = co_author_matrix
num_nodes = adj_mat_csr_sparse.shape[0]
indices = adj_mat_csr_sparse.indices
indptr = adj_mat_csr_sparse.indptr
data = adj_mat_csr_sparse.data
random_walk_length = 100
for t in tqdm(range(num_nodes)):  # t is row index
    for v in indices[indptr[t]:indptr[t + 1]]:  # i.e  possible next ndoes from t
        pi_vx_indices = indices[indptr[v]:indptr[v + 1]]  # i.e  possible next ndoes from v
        pi_vx_values = np.array([alpha(p, q, t, x) for x in pi_vx_indices])
        pi_vx_values = pi_vx_values * data[indptr[v]:indptr[v + 1]]
        # This is eqilvalent to the following
        #         pi_vx_values=[]
        #         for x in pi_vx_indices:
        #             pi_vx=alpha(p,q,t,x)*adj_mat_csr_sparse[v,x]
        #             pi_vx_values.append(pi_vx)
        pi_vx_values[pi_vx_values < 0] = 0.0000001
        pi_vx_values[np.isinf(pi_vx_values)] = 0.0000001
        pi_vx_values[np.isnan(pi_vx_values)] = 0.0000001
        pi_vx_values = pi_vx_values * 1.0 / np.sum(pi_vx_values)
        pi_vx_values[np.isnan(pi_vx_values)] = 1
        # now, we have normalzied transion probabilities for v traversed from t
        # the probabilities are stored as a sparse vector.
        transition[t, v] = (pi_vx_indices, pi_vx_values)
print(pi_vx_values)
start = time.time()
elapsed_time = time.time() - start

proc = 20
pool = mp.Pool(proc)
callback = pool.map(get_random_walk, range(20))
pool.close()

elapsed_time = time.time() - start
print(("elapsed_time:{0}".format(elapsed_time)) + "[sec]")

random_walks = []
for temp in callback:
    random_walks.extend(temp)
del callback
np_random_walks = np.array(random_walks, dtype=np.int32)
del random_walks
np.savez(data_path + dataset + '_rw.npz')

elapsed_time = time.time() - start
print(("elapsed_time:{0}".format(elapsed_time)) + "[sec]")

# tmp = ''
sentences = MySentences(np_random_walks)
model = Word2Vec(size=128, window=5, negative=5, min_count=0)
model.build_vocab(sentences=sentences)
model.train(sentences, epochs=3, total_examples=model.corpus_count, compute_loss=True)
say_vector = model['1']
model.save(data_path + dataset + '_model')
print(say_vector)


