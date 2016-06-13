import pandas as pd
import Preprocessing
import ItemKNN


rating_df = pd.read_csv('data/ratings.dat')
rating_df.columns = ['userId', 'movieId', 'rating', 'time']
rating_df.drop('time', axis=1, inplace=True)
train_df, test_df = Preprocessing.train_test_split(rate_df=rating_df)

train_df.to_csv('data/train.csv', header=True, index=False)
test_df.to_csv('data/test.csv', header=True, index=False)

item_knn_obj = ItemKNN.ItemKNN(_k=10 , rating_file_path='data/train.csv', has_header=True, n_jobs=5)
item_similar = item_knn_obj.compute_similarity_matrix(1, 10, False, False, True)
predicted_rating = item_knn_obj.predict(1, 1, item_similar)
print predicted_rating
