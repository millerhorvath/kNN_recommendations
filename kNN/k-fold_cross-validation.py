import pandas as pd
import Preprocessing
import ItemKNN
import UserKNN
import cPickle as pickle
import time
import os

# Number of neighbours
k = 100

# Folder for data (training dataset, testing dataset, evaluation, similarity matrices and results)
DEF_FOLDER = 'data/1M_ItemKNN_k100/'

rmse_avg = 0.0 # Root Mean Squared Error
mae_avg = 0.0 # Mean Absolute Error
coverage_avg = 0.0 # Coverage

# k-fold cross-validation with 5 subsamples for Item Based
for i in range(5):
    t_start = time.time()
    item_knn_obj = ItemKNN.ItemKNN(_k=k , rating_file_path=DEF_FOLDER+'train/myTrain{}.csv'.format(i), has_header=True, n_jobs=5)
    t_end = time.time()
    
    print("ItemKNN built in {} sec".format(t_end - t_start))
    
    if os.path.exists(DEF_FOLDER+'item_based/similar_matrices/my_item_similar{}.pkl'.format(i)):
        # Loading similarity matrix from file
        t_start = time.time()
        item_similar = item_knn_obj.compute_similarity_matrix(1, k, True, False, True, DEF_FOLDER+'item_based/similar_matrices/my_item_similar{}.pkl'.format(i))
        t_end = time.time()
  		
        print("Similarity Matrix loaded in {} sec".format(t_end - t_start))
    else:
        # Computing similarity matrix
        t_start = time.time()
        item_similar = item_knn_obj.compute_similarity_matrix(1, k, False, False, True)
        t_end = time.time()
        
        print("Similarity Matrix computed in {} sec".format(t_end - t_start))
        
        if not os.path.exists(DEF_FOLDER+'item_based/similar_matrices'):
            os.makedirs(DEF_FOLDER+'item_based/similar_matrices')
            
        pickle.dump(item_similar, open(DEF_FOLDER+'item_based/similar_matrices/my_item_similar{}.pkl'.format(i), 'wb'))
    
    # Computing RMSE, MAE and Coverage
    if not os.path.exists(DEF_FOLDER+'item_based/eval'):
            os.makedirs(DEF_FOLDER+'item_based/eval')
            
    t_start = time.time()
    (rmse, mae, coverage) = item_knn_obj.root_mean_squared_error(DEF_FOLDER+'test/myTest{}.csv'.format(i),
                                True, item_similar, DEF_FOLDER+'item_based/eval/myEval{}.csv'.format(i))
                                
    t_end = time.time()
    
    print("Mean Squared Error computed in {} sec".format(t_end - t_start))
    
    print("MSE: {}".format(rmse))
    print("MAE: {}".format(mae))
    print("Coverage: {}".format(coverage))
    
    rmse_avg += rmse
    mae_avg += mae
    coverage_avg += coverage
    print("")
    
rmse_avg /= 5.0
mae_avg /= 5.0
coverage_avg /= 5.0

# Write results on file
f = open(DEF_FOLDER+'item_based/results.csv', "w")
f.write("Average RMSE,{}\n".format(rmse_avg))
f.write("Average MAE,{}\n".format(mae_avg))
f.write("Average Coverage,{}\n".format(coverage_avg))
f.close()

print("Average RMSE: {}".format(rmse_avg))
print("Average MAE: {}".format(mae_avg))
print("Average Coverage: {}".format(coverage_avg))
print("")

rmse_avg = 0.0 # Root Mean Squared Error
mae_avg = 0.0 # Mean Absolute Error
coverage_avg = 0.0 # Coverage


# k-fold cross-validation with 5 subsamples for User Based
for i in range(5):
    t_start = time.time()
    user_knn_obj = UserKNN.UserKNN(_k=k , rating_file_path=DEF_FOLDER+'train/myTrain{}.csv'.format(i), has_header=True, n_jobs=5)
    t_end = time.time()
    
    print("UserKNN built in {} sec".format(t_end - t_start))
    
    if os.path.exists(DEF_FOLDER+'user_based/similar_matrices/my_user_similar{}.pkl'.format(i)):
        # Loading similarity matrix from file
        t_start = time.time()
        user_similar = user_knn_obj.compute_similarity_matrix(1, k, True, False, True, DEF_FOLDER+'user_based/similar_matrices/my_user_similar{}.pkl'.format(i))
        t_end = time.time()
  		
        print("Similarity Matrix loaded in {} sec".format(t_end - t_start))
    else:
        # Computing similarity matrix
        t_start = time.time()
        user_similar = user_knn_obj.compute_similarity_matrix(1, k, False, False, True)
        t_end = time.time()
        
        print("Similarity Matrix computed in {} sec".format(t_end - t_start))
        
        if not os.path.exists(DEF_FOLDER+'user_based/similar_matrices'):
            os.makedirs(DEF_FOLDER+'user_based/similar_matrices')
            
        pickle.dump(user_similar, open(DEF_FOLDER+'user_based/similar_matrices/my_user_similar{}.pkl'.format(i), 'wb'))
    
    # Computing RMSE, MAE and Coverage
    if not os.path.exists(DEF_FOLDER+'user_based/eval'):
            os.makedirs(DEF_FOLDER+'user_based/eval')
            
    t_start = time.time()
    (rmse, mae, coverage) = user_knn_obj.root_mean_squared_error(DEF_FOLDER+'test/myTest{}.csv'.format(i),
                                True, user_similar, DEF_FOLDER+'user_based/eval/myEval{}.csv'.format(i))
                                
    t_end = time.time()
    
    print("Mean Squared Error computed in {} sec".format(t_end - t_start))
    
    print("MSE: {}".format(rmse))
    print("MAE: {}".format(mae))
    print("Coverage: {}".format(coverage))
    
    rmse_avg += rmse
    mae_avg += mae
    coverage_avg += coverage
    print("")
    
rmse_avg /= 5.0
mae_avg /= 5.0
coverage_avg /= 5.0

# Write results on file
f = open(DEF_FOLDER+'user_based/results.csv', "w")
f.write("Average RMSE,{}\n".format(rmse_avg))
f.write("Average MAE,{}\n".format(mae_avg))
f.write("Average Coverage,{}\n".format(coverage_avg))
f.close()

print("Average RMSE: {}".format(rmse_avg))
print("Average MAE: {}".format(mae_avg))
print("Average Coverage: {}".format(coverage_avg))
print("")
