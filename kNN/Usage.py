import pandas as pd
import Preprocessing
import ItemKNN
import cPickle as pickle

# rating_df = pd.read_csv('data/ratings.dat')
# rating_df.columns = ['userId', 'movieId', 'rating', 'time']
# rating_df.drop('time', axis=1, inplace=True)
# train_df, test_df = Preprocessing.train_test_split(rate_df=rating_df)
#
# train_df.to_csv('data/train.csv', header=True, index=False)
# test_df.to_csv('data/test.csv', header=True, index=False)

k = 100
item_knn_obj = ItemKNN.ItemKNN(_k=k , rating_file_path='data/train.csv', has_header=True, n_jobs=5)

item_similar = item_knn_obj.compute_similarity_matrix(1, k, False, False, True)
pickle.dump(item_similar, open('data/item_similar.pkl', 'wb'))

# item_similar = pickle.load(open('data/item_similar.pkl', 'rb'))

print item_similar[item_knn_obj.movieId_to_idx[920]]

predicted_rating = item_knn_obj.predict(1, 914, item_similar) # actual is 3
print predicted_rating
