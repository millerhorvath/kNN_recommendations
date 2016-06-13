import numpy as np
import cPickle as pkl
import random
import time
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
# import post_diversification
# import evaluate


"""
This is a ItemKNN method.
rating data is only based on training. There is no test data here.
We just get training. Compute the item-item similarity matrix.
and then get the recommendations(items)
"""


def find_neighbors_users_par(movie_index, item_similar, user_num, item_user_matrix, k):
    sum_vec_users = np.zeros((user_num), dtype=float)
    count_neighbors = 0
    for sim_list in item_similar[movie_index]:
        # sim_list = [[user_id, simVal], [...]]
        count_neighbors += 1

        # neighbor rating for this movieId
        nei_movie_index = sim_list[0]
        # nei_similarity = sim_list[1]
        # nei_movie_index = movieId_to_idx[nei_movie_id]
        sum_vec_users += item_user_matrix[nei_movie_index]

        if count_neighbors == k:
            break

    # non-zero user indices of neighbors
    nonzero_user_index = list(sum_vec_users.nonzero()[0])
    return nonzero_user_index


def predict_par(item_similar, movie_index, user_index, item_user_matrix, k):
    sum_rates = 0.0
    sum_similarity = 0.0
    count_neighbors = 0
    for sim_list in item_similar[movie_index]:
        # sim_list = [[user_id, simVal], [...]]
        count_neighbors += 1

        # neighbor rating for this movieId
        nei_movie_index = sim_list[0]
        nei_similarity = sim_list[1]
        # nei_movie_index = movieId_to_idx[nei_movie_index]
        neighbor_rate = item_user_matrix[nei_movie_index, user_index]
        # if float(neighbor_rate) <= 5.1:
        #     print type(neighbor_rate)
        #     print 'rating value', neighbor_rate
        #     raise Exception('Suspicious neighbor rate; out of bound [1,5] !!')

        if neighbor_rate > 0:
            sum_rates += neighbor_rate * nei_similarity
            sum_similarity += nei_similarity

        if count_neighbors == k:
            break

    pred_rate = 0
    if sum_rates > 0:
        pred_rate = sum_rates / float(sum_similarity)

    # print 'pred_rate', pred_rate

    # if pred_rate > 5.1:
    #     # userId = idx_to_userId[user_index]
    #     raise Exception('prediction rates should NOT be larger than 5.0!! '
    #                     'but for user=%d and item=%d the prediction rate=%.3f.' % (user_index, movie_id, pred_rate))

    return pred_rate


def recommend_one(movie_index, top_n, item_similar, user_num, item_user_matrix, k):


    # print 'user id', user_id
    # movie_index = movieId_to_idx[movie_id]
    predictions = []

    # item_user_matrix[:, user_index]

    # find non-zero movie rates for the neighbors of this user.
    # Not all items are going to get a rate(non-zero prediction). Only those items that the neighbors
    # of this user has seen are considered as non-zero prediction rates.
    neighbors_users_index = find_neighbors_users_par(movie_index, item_similar,
                                                      user_num, item_user_matrix, k)
    # userId, user_similar, movies_num, userId_to_idx, user_item_matrix, k
    # neighbors_movies_index = []

    # this user's mean rating for all items
    movie_mean_rate = np.mean(item_user_matrix[movie_index])

    # movie_rates = np.random.normal(movie_mean_rate, 0.7, user_num)

    movie_rates = np.zeros((user_num))
    movie_rates.fill(movie_mean_rate)

    # if len(neighbors_users_index) < top_n:
    #     print 'Users who rated this movie by neighbors of this movie are less than top_n required !!!'
    #     print 'top_n = %d,\t rated_users = %d' % (top_n, len(neighbors_users_index))

    for user_index in neighbors_users_index:
        # movie_id = idx_to_movieId[movie_index]

        if item_user_matrix[movie_index, user_index] == 0:
            pred_rate = predict_par(item_similar, movie_index, user_index, item_user_matrix, k)

            # pred_rate = random.randint(1, 5)

            # predictions.append([movie_id, pred_rate])
            movie_rates[user_index] = pred_rate


    # copying movies from numpy array to sort and pick top_n ratings
    # for user_index in range(len(all_users)):
    #     user_id = idx_to_userId[user_index]
    #     predictions.append([user_id, movie_rates[user_index]])

    # predictions.sort(key=lambda x: x[1], reverse=True)
    # return [movie_id, [user[0] for user in predictions[:top_n]]]
    return [movie_index, movie_rates]


class ItemKNN():

    def __init__(self, _k, rating_file_path, has_header):
        """
        :param _k: k nearest neighbors
        :param rating_file_path: file name + path for the "user,item,rating" file; This is only training data
        :param has_header: Does the rating file has header
        """

        print 'ItemKNN init'
        self.data_file_path = 'recs_data_2/'

        """ rate_df is ONLY the training part. Not test set!! """
        # self.rate_df = _rate_df

        self.k = _k
        self.has_header = has_header

        # self.item_user_matrix = None
        self.item_user_matrix_low = None

        self.data_path = 'Diversity_2/item_knn/'
        self.item_similar_file_path = self.data_path + 'item_similarity_matrix.pkl'
        self.user_recs_file_path = self.data_path + 'user_recs.pkl'
        self.item_user_matrix_file_path = self.data_path + 'item_user_matrix'

        # Creating item-user matrix
        self.__read_data_into_matrix__(rating_file_path)



    def __read_data_into_matrix__(self, file_path):

        f = open(file_path, 'r')
        user = []
        item = []
        if self.has_header:
            f.readline()
        for line in f:
            spline = line.split(',')
            user.append(int(spline[0]))
            item.append(int(spline[1]))

        self.all_users = list(set(user))
        self.all_movies = list(set(item))

        self.user_num = len(self.all_users)
        self.movies_num = len(self.all_movies)
        print 'users num:', self.user_num
        print 'items num:', self.movies_num

        # map of index to movieId and vice versa
        self.idx_to_movieId = {}
        self.movieId_to_idx = {}
        for i in range(self.movies_num):
            self.idx_to_movieId[i] = self.all_movies[i]
            self.movieId_to_idx[self.all_movies[i]] = i

        self.idx_to_userId = {}
        self.userId_to_idx = {}
        for i in range(self.user_num):
            self.idx_to_userId[i] = self.all_users[i]
            self.userId_to_idx[self.all_users[i]] = i


        self.item_user_matrix = np.zeros((self.movies_num, self.user_num), dtype=float)
        f = open(file_path, 'r')
        if self.has_header:
            f.readline()
        for line in f:
            spline = line.split(',')
            userIndex = self.userId_to_idx[int(spline[0])]
            movieIndex = self.movieId_to_idx[int(spline[1])]
            self.item_user_matrix[movieIndex, userIndex] = float(spline[2])


        # print 'Done!!'


    def item_similarity_sklearn(self, top_n):
        # minkowski
        nbrs = NearestNeighbors(n_neighbors=top_n+1, algorithm='ball_tree', metric='euclidean', n_jobs=10).\
            fit(self.item_user_matrix)
        # indices for nearest neighbors and their distances
        distances, indices = nbrs.kneighbors(self.item_user_matrix)
        item_similars = {}
        for item_idx in range(len(distances)):
            # item_id = self.idx_to_movieId[item_idx]
            sim_list = []
            for nei in range(1, top_n+1):
                # nei_id = self.idx_to_movieId[indices[item_idx][nei]]
                nei_index = indices[item_idx][nei]
                sim = distances[item_idx][nei]
                sim_list.append([nei_index, sim])
            item_similars[item_idx] = sim_list[:]

        pkl.dump(item_similars, open(self.item_similar_file_path, 'wb'))


    def compute_similarity_matrix(self, low_dim, top_n, is_sim_infile, is_use_low_dim, is_sklearn_kNN_sim):
        """
        Usage:
        item_similar = compute_similarity_matrix(1, top_n, False, False, True)

        :param low_dim:
            indicates the size of low dimension for item-user matrix .The low dimension
            matrix is used to compute item-item similarities.

        :param top_n: compute and save the top_n similar users in the matrix(actually it's a list).

        :param is_sim_infile:
            Is user similarity file saved in file, if true then this method will load it from file.
        :return: item-item similarity matrix
        """

        if not is_sim_infile:
            # print 'Loading item-user matrix(numpy array)...'
            # self.item_user_matrix = self.create_item_user_matrix()

            if is_use_low_dim:
                print 'get a lower dimension of item-user matrix for similarity using SVD...'
                raise Exception("Implementation of SVD for dimensionality reduction has removed!!")
                # self.item_user_matrix_low = low_dim_similarity.user_svd(self.item_user_matrix,
                #                                                         low_dim)
            else:
                self.item_user_matrix_low = self.item_user_matrix

            if is_sklearn_kNN_sim:
                print 'Computing sk-learn kNN similarity...'
                self.item_similarity_sklearn(top_n)
                # print 'Done!!'

            else:
                print 'Computing item-item similarity...'
                raise Exception('THERE IS NO IMPLEMENTATION OF ITEM_SIMILARITY (removed)!!')
                # self.user_similarity(top_n)

        print 'Loading user-user similarity from file...'
        raise Exception('Nothing saved!! We do NOT save anything in file in this implementation.')
        # item_similar = pkl.load(open(self.item_similar_file_path, 'rb'))

        return item_similar


    def recommend_all(self, top_n, item_similar, rec_to_file_name):
        print 'Item recommendation...'

        """
            Get top_n recommendations for all users
        :param top_n:
        :param rec_rec_to_file_name:
            file_name with path where the recommendations will be write on.
            You can use this file later to get the recommendations for each user.
            Read this file using cPickle.read(file_name). This will give you a dict. The keys are user_ids
            and the values are list of recommendations that are item_ids.

        :return:
        """
        # print 'Loading user-item matrix(numpy array)...'
        # self.item_user_matrix = np.load(self.item_user_matrix_file_path + '.npy')

        # self.user_item_matrix = self.create_user_item_matrix()

        # mean ratings based on user-item index
        # mean_user_item_ratings = np.mean(self.user_item_matrix, axis=0)

        user_recs = {}
        # for user_id in self.all_users:
        _all_users = self.all_users
        # backend = "threading"
        movie_recs_par = Parallel(n_jobs=15, backend="threading", verbose=0, max_nbytes="900M")\
            (delayed(recommend_one)(movie_index, top_n, item_similar,
                                    self.user_num,
                                    self.item_user_matrix,
                                    self.k)
                           for movie_index in range(self.movies_num))


        # make item_user predictions matrix
        item_user_pred = np.zeros(self.item_user_matrix.shape, dtype=float)


        pkl.dump(movie_recs_par, open(self.data_path + 'movie_recs.pkl', 'wb'))
        for movie_pred in movie_recs_par:
            # print 'movie_pred[1]', movie_pred[1]
            # print np.array(movie_pred[1])
            item_user_pred[movie_pred[0]] = np.array(movie_pred[1])

        for user_index in range(self.user_num):
            items_index = np.argsort(item_user_pred[:, user_index])[::-1][:top_n]
            user_id = self.idx_to_userId[user_index]
            top_item_ids = []
            for item_index in items_index:
                item_id = self.idx_to_movieId[item_index]
                top_item_ids.append(item_id)

            user_recs[user_id] = top_item_ids


        # for r in user_recs_par:
        #     user_recs[r[0]] = r[1]
        pkl.dump(user_recs, open(self.user_recs_file_path, 'wb'))
        return user_recs

