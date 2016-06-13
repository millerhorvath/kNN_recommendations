import pandas as pd
import numpy as np
from numpy import linalg as LA
import cPickle
import random
from scipy.stats import rv_discrete
from sklearn import preprocessing


def train_test_split(rate_df):
    print 'train_test split...'
    all_users = pd.unique(rate_df.userId.ravel())
    all_movies = pd.unique(rate_df.movieId.ravel())

    movieIdx = 0
    movieId_to_Index = {}
    movieIndex_to_Id = {}
    for movieId in all_movies:
        movieId_to_Index[movieId] = movieIdx
        movieIndex_to_Id[movieIdx] = movieId
        movieIdx += 1

    # defining the inverse of popularity sample proba space
    # We should use index here for items because we are going to sample
    max_movieId = max(all_movies)
    item_count = np.zeros((1, len(all_movies)), dtype=int)

    for _, row in rate_df.iterrows():
        movie_id = row['movieId']
        item_count[0, movieId_to_Index[movie_id]] += 1

    item_count = item_count.flatten()
    # print 'item count', item_count
    # for i in item_count:
    #     print i,

    # total_sample contains movieINDICES not IDs, so we need to the dictionaries.
    # total_sample = np.array([1.0 / (i**3) for i in item_count]).reshape(1, -1)
    # total_sample = np.array([1.0 / (i) for i in item_count]).reshape(1, -1)

    total_sample = np.array([1.0 / len(item_count) for i in item_count]).reshape(1, -1)

    # print
    # for i in total_sample[0]:
    #     print i,
    # return

    total_sample = preprocessing.normalize(total_sample, norm='l1').flatten()
    # folder = 'dist_data_small/all_users_range('
    # # load only those userIDs that we've been able to get their ILD score.
    # all_users = pickle.load(open(folder+('%d,%d).pkl' % (0, 300)), 'rb'))
    # all_users += pickle.load(open(folder+('%d,%d).pkl' % (300, 700)), 'rb'))

    trains = []
    tests = []
    test_split_percent = .2
    k_fold = 5
    '''
    for each cross_validation we need to concat different users' train into a flat list that we are going to use it
    as the rating matrix for train. and another for test.
    So, train_lists_cv = [[all train ratings for cv_1], [all train ratings for cv_2], ...]
    and for each train there is a test set with the same index.
    test_lists_cv = [[all test ratings for cv_1], ...]
    '''
    print 'len users', len(all_users)

    train = []
    test = []
    for user in all_users:
        print 'user', user
        X = rate_df[rate_df['userId'] == user][['userId', 'movieId', 'rating']]
        # if user % user_test_num != 0:
        # # if True == False:
        #     for _, row in X.iterrows():
        #         train.append([row['userId'], row['movieId'], row['rating']])
        # else:

        # print '-'*50
        # print 'len of X for one user', len(X)
        # print 'X', X
        # print '-'*50

        n_test_case = .2 * len(X)
        test_cases = []

        count = 0
        print n_test_case

        # items seen by this user
        items_seen = X['movieId'].values.tolist()
        small_sample_proba = []
        for movieId in items_seen:
            movieIdx = movieId_to_Index[movieId]
            small_sample_proba.append(total_sample[movieIdx])

        # normalize the small sample
        small_sample_proba = np.array(small_sample_proba).reshape(1, -1)
        small_sample_proba = preprocessing.normalize(small_sample_proba, norm='l1').flatten()

        # in samples we have movieIDs not INDEX
        sample = rv_discrete(values=(items_seen, small_sample_proba)).rvs(size=n_test_case)
        test_cases = list(set(sample))
        while len(test_cases) < n_test_case:
            sample = rv_discrete(values=(items_seen, small_sample_proba)).rvs(size=n_test_case)
            test_cases = list(set(test_cases + list(sample)))
            count += 1

            if count > 500:
                break
            # print small_sample_proba
            # print sample
            # print test_cases
            # print n_test_case
            # print '-'*20
        for _, row in X.iterrows():
            if row['movieId'] in test_cases:
                test.append([row['userId'], row['movieId'], row['rating']])
            else:
                train.append([row['userId'], row['movieId'], row['rating']])
        # print '='*20

        # while len(test_cases) < n_test_case:
        #     count += 1
        #     start = timeit.default_timer()
        #     test_case = draw_item(movie_select_sample, X['movieId'].values.tolist())
        #     end = timeit.default_timer()
        #     print 'draw', end - start
        #
        #     # print 'test drawn', test_case
        #     if test_case not in test_cases:
        #         # start = timeit.default_timer()
        #         movie_select_sample = dec_sample_proba(movie_select_sample, test_case)
        #         # end = timeit.default_timer()
        #         # print 'dec sample', end - start
        #         test_cases.append(test_case)

        # print test_cases
        # print 'count', count
        # print '-'*20
        # test cases are movie Ids, so we need to separate train and test tuples(userId, movieId, rating)


        # if user > 11:
        #     return

        # print '-'*100
        # print train
        # print '-' * 100
        # print test

    # print '='*200
    # print 'Movie select sample:'
    # print movie_select_sample
    #


    train_df = pd.DataFrame(train, columns=['userId', 'movieId', 'rating'])
    test_df = pd.DataFrame(test, columns=['userId', 'movieId', 'rating'])
    # saving train, test into file
    # train_df.to_csv(data_path+train_fn, index=False)
    # test_df.to_csv(data_path + test_fn, index=False)

    return train_df, test_df
