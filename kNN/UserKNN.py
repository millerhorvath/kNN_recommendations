# import pandas as pd
import numpy as np
import cPickle as pkl
import random
import time
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
import math
# import post_diversification
# import evaluate


"""
This is a UserKNN method.
rating data is only based on training. There is no test data here.
We just get training. Compute the user-user similarity matrix.
and then get the recommendations(items)
"""


def find_neighbors_items_par(userId, user_similar, movies_num, user_item_matrix, k):
    sum_vec_items = np.zeros((movies_num), dtype=float)
    count_neighbors = 0
    for sim_list in user_similar[userId]:
        # sim_list = [[user_id, simVal], [...]]
        count_neighbors += 1

        # neighbor rating for this movieId
        nei_user_id = sim_list[0]
        # nei_similarity = sim_list[1]
        #nei_user_index = userId_to_idx[nei_user_id]
        sum_vec_items += user_item_matrix[nei_user_id]

        if count_neighbors == k:
            break

    # non-zero movie indices of neighbors
    nonzero_movie_index = list(sum_vec_items.nonzero()[0])
    return nonzero_movie_index


def predict_par(user_similar, movie_index, user_index, user_item_matrix, k):
    sum_rates = 0.0
    sum_similarity = 0.0
    count_neighbors = 0
    for sim_list in user_similar[user_index]:
        # sim_list = [[user_id, simVal], [...]]
        count_neighbors += 1

        # neighbor rating for this movieId
        nei_user_id = sim_list[0]
        nei_similarity = sim_list[1]
        #nei_user_index = userId_to_idx[nei_user_id]
        neighbor_rate = user_item_matrix[nei_user_id, movie_index]
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

    #if pred_rate > 5.1:
    #    movieId = idx_to_movieId[movie_index]
    #    raise Exception('prediction rates should NOT be larger than 5.0!! '
    #                    'but for user=%d and item=%d the prediction rate=%.3f.' % (userId, movieId, pred_rate))

    return pred_rate


def recommend_one(user_index, top_n, user_similar, movies_num, user_item_matrix, k):
                #idx_to_movieId, k, all_movies, movieId_to_idx):

    # print 'user id', user_id
    # user_index = userId_to_idx[user_id]
    predictions = []



    # find non-zero movie rates for the neighbors of this user.
    # Not all items are going to get a rate(non-zero prediction). Only those items that the neighbors
    # of this user has seen are considered as non-zero prediction rates.
    neighbors_movies_index = find_neighbors_items_par(user_index, user_similar,
                                                       movies_num, user_item_matrix, k)
    # userId, user_similar, movies_num, userId_to_idx, user_item_matrix, k
    # neighbors_movies_index = []

    # this user's mean rating for all items
    user_mean_rate = np.mean(user_item_matrix[user_index])



    user_rates = np.zeros(movies_num)
    user_rates.fill(user_mean_rate)

    # if len(neighbors_movies_index) < top_n:
    #     print 'Movies rated by neighbors of this user are less than top_n required !!!'
    #     print 'top_n = %d,\t rated_movies = %d' % (top_n, len(neighbors_movies_index))

    for movie_index in neighbors_movies_index:
        # movie_id = idx_to_movieId[movie_index]

        if user_item_matrix[user_index, movie_index] == 0:
            pred_rate = predict_par(user_similar, movie_index, user_index, user_item_matrix, k)

            # predictions.append([movie_id, pred_rate])
            user_rates[movie_index] = pred_rate


    ## copying movies from numpy array to sort and pick top_n ratings
    #for movie_index in range(len(all_movies)):
    #    movie_id = idx_to_movieId[movie_index]
    #    predictions.append([movie_id, user_rates[movie_index]])


    # for those movies that are not in user neighbors, use average rating of all users
    # for movie_id in all_movies:
    #     movie_index = movieId_to_idx[movie_id]
    #     if movie_index not in neighbors_movies_index:
    #         predictions.append([movie_id, mean_user_item_ratings[movie_index]])

    #predictions.sort(key=lambda x: x[1], reverse=True)

    # user_recs[user_id] = [movie[0] for movie in predictions[:top_n]]
    #return [user_id, [movie[0] for movie in predictions[:top_n]]]
    return [user_index, user_rates]


class UserKNN():

    def __init__(self, _k, rating_file_path, has_header, n_jobs):
        """
            :param _k: k nearest neighbors
            :param rating_file_path: file name + path for the "user,item,rating" file; This is only training data
            :param has_header: Does the rating file has header
        """

        print 'UserKNN init'
        # self.data_file_path = 'recs_data_2/'

        """ rate_df is ONLY the training part. Not test set!! """
        # self.rate_df = _rate_df
        self.k = _k
        self.n_jobs = n_jobs
        self.has_header = has_header
        
        self.cov_error = 0

        self.user_item_matrix_low = None

        # self.data_path = 'Diversity_2/user_knn/'
        # self.user_similar_file_path = self.data_path + 'user_similarity_matrix.pkl'
        # self.user_recs_file_path = self.data_path + 'user_recs.pkl'
        # self.user_item_matrix_file_path = self.data_path + 'user_item_matrix'

        self.__read_data_into_matrix__(rating_file_path)


    def __read_data_into_matrix__(self, file_path):

        f = open(file_path, 'r')
        user = []
        item = []
        if self.has_header:
            f.readline()
            
	self.all_movies_avg = 0.0
	count = 0
            
        for line in f:
            spline = line.split(',')
            user.append(int(spline[0]))
            item.append(int(spline[1]))
            self.all_movies_avg += float(spline[2])
            count = count + 1
            
        self.all_movies_avg /= float(count)

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

        self.user_item_matrix = np.zeros((self.user_num, self.movies_num), dtype=float)
        
	f = open(file_path, 'r')
        if self.has_header:
            f.readline()
            
        for line in f:
            spline = line.split(',')
            userIndex = self.userId_to_idx[int(spline[0])]
            movieIndex = self.movieId_to_idx[int(spline[1])]
            self.user_item_matrix[userIndex, movieIndex] = float(spline[2])

        # np.save(self.user_item_matrix_file_path, self.user_item_matrix)


    def user_similarity_sklearn(self, top_n):
        # minkowski
        nbrs = NearestNeighbors(n_neighbors=top_n+1, algorithm='ball_tree', metric='euclidean', n_jobs=self.n_jobs).\
            fit(self.user_item_matrix)
        # indices for nearest neighbors and their distances
        distances, indices = nbrs.kneighbors(self.user_item_matrix)
        user_similars = {}
        for user_idx in range(len(distances)):
            #user_id = self.idx_to_userId[user_idx]
            sim_list = []
            for nei in range(1, top_n+1):
                #nei_id = self.idx_to_userId[indices[user_idx][nei]]
                nei_index = indices[user_idx][nei]
                sim = distances[user_idx][nei]
                sim_list.append([nei_index, sim])
            user_similars[user_idx] = sim_list[:]

        # don't save user_similar because we may run several instances of these in parallel.
        # only return these values
        # pkl.dump(user_similars, open(self.user_similar_file_path, 'wb'))

        return user_similars


    def predict(self, user_id, movie_id, user_similar):
        movie_error_flag = False
        user_error_flag = False
	
	# Dealing with movie coverage error
	try:
	   movie_index = self.movieId_to_idx[movie_id]
	except KeyError:
	   movie_error_flag = True
	
	
	
	# Dealing with user coverage error
	try:
	   user_index = self.userId_to_idx[user_id]
	except:
	   movie_error_flag = True
	
	
	
	if movie_error_flag == True:
	   self.cov_error = self.cov_error + 1
	   #try:
	   #    print 'Movie Coverage Error'
	   #except:
	   #    pass
	       
	   return self.all_movies_avg
	   #return 'Movie Coverage Error'
	elif user_error_flag == True:
	   self.cov_error = self.cov_error + 1
	   #try:
	   #    print 'User Coverage Error'
	   #except:
	   #    pass
	       
	   return self.all_movies_avg
	   #return 'User Coverage Error'
	else:
	   #try:
	   #    print 'movie_index', movie_index
	   #    print 'user index', user_index
	   #except:
	   #    pass
	   
	   
	   rating = predict_par(user_similar, movie_index, user_index, self.user_item_matrix, self.k)
	   if rating == 0:
	       self.cov_error = self.cov_error + 1
	       
	       # mean ratings of this item based on user-item index
	       mean_rating_item = np.mean(self.user_item_matrix[movie_index][self.user_item_matrix[movie_index].nonzero()[0]])
	       
	       # print self.user_item_matrix[self.user_item_matrix[movie_index].nonzero()]
	       #print 'This is average rating.'
	       return mean_rating_item
	   return rating
        
    def root_mean_squared_error(self, test_file_path, has_header, user_similar, out_file_eval = ""):
        print ("Computing Root Mean Squared Error...")
        
        # Reading Testing Data Set File
        f = open(test_file_path, "r")
        
        if has_header:
            f.readline()
        
        inFile = f.readlines()
        
        f.close()
        
        
        rmse_sum = 0.0 # sum of the squared difference between observed and predicted ratings
        mae_sum = 0.0 # sum of the posite diference between observed and predicted ratings
        n = 0 # conunting number of ratings in training set
        
		# Predicting a rating for all entries into testing dataset
        for i in range(len(inFile)):
            inFile[i] = inFile[i].split(",")
            
            # Converting data to the proper data type
            for j in range(len(inFile[i]) - 1):
                inFile[i][j] = int(inFile[i][j])
            inFile[i][2] = float(inFile[i][2])
            
            # Getting prediction given a user and an item
            inFile[i].append(self.predict(inFile[i][0], inFile[i][1], user_similar))
            
            # RMSE summing
            rmse_sum += ((inFile[i][3] - inFile[i][2]) * (inFile[i][3] - inFile[i][2]))
            
            # MAE summing
            mae_sum += math.fabs(inFile[i][3] - inFile[i][2])
            
            # counting
            n = n + 1
        
        rmse = math.sqrt(rmse_sum / float(n)) # RMSE
        
        mae = mae_sum / float(n) # MAE
        
        coverage = (float(n) - float(self.cov_error)) / float(n) # Coverage
        
	# Writing fold evaluation on file
	if out_file_eval != "":
            f = open(out_file_eval, "w")
  		
            f.write("RMSE:,{}\n".format(rmse))
            f.write("MAE:,{}\n".format(mae))
            #print ("Number of inaccurate predictions: {}".format(self.cov_error))
            #print ("Number of prediction attempts: {}".format(n))
            f.write("Coverage:,{}\n".format(coverage))
  		
            f.write("userId,movieId,rating, pred_rating\n")
            
            for i in inFile:
                for j in range(len(i)):
                    i[j] = str(i[j])
                f.write(",".join(i))
                f.write("\n")
            
            f.close()
        
        return (rmse, mae, coverage)



    # def predict(self, user_similar, userId, movie_index):
    #
    #     sum_rates = 0.0
    #     sum_similarity = 0.0
    #     count_neighbors = 0
    #     for sim_list in user_similar[userId]:
    #         # sim_list = [[user_id, simVal], [...]]
    #         count_neighbors += 1
    #
    #         # neighbor rating for this movieId
    #         nei_user_id = sim_list[0]
    #         nei_similarity = sim_list[1]
    #         nei_user_index = self.userId_to_idx[nei_user_id]
    #         neighbor_rate = self.user_item_matrix[nei_user_index, movie_index]
    #         # if float(neighbor_rate) <= 5.1:
    #         #     print type(neighbor_rate)
    #         #     print 'rating value', neighbor_rate
    #         #     raise Exception('Suspicious neighbor rate; out of bound [1,5] !!')
    #
    #         if neighbor_rate > 0:
    #             sum_rates += neighbor_rate * nei_similarity
    #             sum_similarity += nei_similarity
    #
    #         if count_neighbors == self.k:
    #             break
    #
    #     pred_rate = 0
    #     if sum_rates > 0:
    #         pred_rate = sum_rates / float(sum_similarity)
    #
    #     # print 'pred_rate', pred_rate
    #
    #     if pred_rate > 5.1:
    #         movieId = self.idx_to_movieId[movie_index]
    #         raise Exception('prediction rates should NOT be larger than 5.0!! '
    #                         'but for user=%d and item=%d the prediction rate=%.3f.' % (userId, movieId, pred_rate))
    #
    #     return pred_rate


    def compute_similarity_matrix(self, low_dim, top_n, is_sim_infile, is_use_low_dim, is_sklearn_kNN_sim, user_similar_file_path = ""):
        """
        :param low_dim:
            indicates the size of low dimension for user-item matrix .The low dimension
            matrix is used to compute user-user similarities.

        :param top_n: compute and save the top_n similar users in the matrix(actually it's a list).

        :param is_sim_infile:
            Is user similarity file saved in file, if true then this method will load it from file.
        :return:
        """

        #user_similar = None

        if not is_sim_infile:
            # print 'Creating user-item matrix(numpy array)...'
            # self.user_item_matrix = self.create_user_item_matrix()

            if is_use_low_dim:
                print 'get a lower dimension of user-item matrix for similarity using SVD...'
                raise Exception("Implementation of SVD for dimensionality reduction has removed!!")
                # self.user_item_matrix_low = low_dim_similarity.user_svd(self.user_item_matrix,
                #                                                         low_dim)
            else:
                self.user_item_matrix_low = self.user_item_matrix

            if is_sklearn_kNN_sim:
                print 'Computing sk-learn kNN similarity...'
                user_similar = self.user_similarity_sklearn(top_n)
                # print 'Done!!'

            else:
                print 'Computing user-user similarity...'
                raise Exception('No implementation for user-user similarity!! use sklearn kNN!!')
                # self.user_similarity(top_n)
        else:
            print 'Loading user-user similarity from file...'
            #raise Exception('Nothing saved!! We do NOT save anything in file in this implementation.')
            user_similar = pkl.load(open(user_similar_file_path, 'rb'))

        return user_similar


    def find_neighbors_items(self, userId, user_similar):

        sum_vec_items = np.zeros((self.movies_num), dtype=float)
        count_neighbors = 0
        for sim_list in user_similar[userId]:
            # sim_list = [[user_id, simVal], [...]]
            count_neighbors += 1

            # neighbor rating for this movieId
            nei_user_id = sim_list[0]
            # nei_similarity = sim_list[1]
            nei_user_index = self.userId_to_idx[nei_user_id]
            sum_vec_items += self.user_item_matrix[nei_user_index]

            if count_neighbors == self.k:
                break

        # non-zero movie indices of neighbors
        nonzero_movie_index = list(sum_vec_items.nonzero()[0])
        return nonzero_movie_index


    def recommend_all(self, top_n, user_similar, rec_to_file_name):
        print 'Item recommendation...'

        """
            Get top_n recommendations for all users
        :param top_n:
        :return:
        """
        # print 'Loading user-item matrix(numpy array)...'
        # self.user_item_matrix = np.load(self.user_item_matrix_file_path + '.npy')
        # self.user_item_matrix = self.create_user_item_matrix()

        # mean ratings based on user-item index
        # mean_user_item_ratings = np.mean(self.user_item_matrix, axis=0)

        user_recs = {}
        # for user_id in self.all_users:
        _all_users = self.all_users
        # backend = "threading"
        user_recs_par = Parallel(n_jobs=self.n_jobs, backend="threading", verbose=0, max_nbytes="100M")\
            (delayed(recommend_one)(user_id, top_n, user_similar,
                                                                    self.movies_num, self.userId_to_idx,
                                                                    self.user_item_matrix,
                                                                    self.idx_to_movieId,
                                                                    self.k, self.all_movies,
                                                                    self.movieId_to_idx)
                           for user_id in _all_users)

        for r in user_recs_par:
            user_recs[r[0]] = r[1]
        pkl.dump(user_recs, open(rec_to_file_name, 'wb'))
        return user_recs

