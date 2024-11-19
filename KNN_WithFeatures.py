'''
KNN Collaborative Filtering Recommender System with item features

For more info please see the demo file for the KNN system, it contains alot of information. This code contains the KNN
recommender system utilizing user and item interaction as well as adding items features to the vector, to predict items 
the users might like aswell. Adding more context to the vector, could in some cases improve the accuracy of the
predicitons. Both these systems performance can be viewed in the demo file.
'''

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from Utils.distance import Distance


#simple example of how to add features and get feature matrix
def get_features(data):
    df = data[['itemID', 'itemTags']]
    df.columns = ['itemID', 'itemTags']
    one_hot = df['itemTags'].str.get_dummies("|")
    return pd.concat([df,one_hot],axis=1).drop(columns='itemTags', axis=0)



class KNN_CF_ITEM_FEATURES:

    def __init__(self, userField, itemField, valueField, featurematrix, type='regression'):
        self.userField = userField
        self.itemField = itemField
        self.valueField = valueField
        self.feature_matrix = featurematrix
        self.predict_type = type
        self.set_distance_metrics()
        self.set_n_recommendations_and_k()
        
    def set_distance_metrics(self, primary_metric = 'cosine', prediction_metric = 'manhattan'):
        self.primary_distance = primary_metric
        self.prediction_metric = prediction_metric
    
    def set_n_recommendations_and_k(self,n=10, k=10):
        self.n_neighbors = n
        self.k = k

    def create_matrix(self, data):
        ''' As with the other system this function creates some usefull maps. However
        for this system it also includes combining the item vectors with the
        normal rating vectors and storing them in a combined matrix.'''

        N = data[self.userField].nunique()
        M = data[self.itemField].nunique()
        user_list = np.unique(data[self.userField])
        item_list = np.unique(data[self.itemField])
        self.user_to_index = dict(zip(user_list, range(0, N)))
        self.item_to_index = dict(zip(item_list, range(0,M)))
        self.index_to_user = dict(zip(range(0,N), user_list))
        self.index_to_item = dict(zip(range(0,M), item_list))
        user_index = [self.user_to_index[i] for i in data[self.userField]]
        item_index = [self.item_to_index[i] for i in data[self.itemField]]
        self.rating_matrix = csr_matrix((data[self.valueField], (item_index, user_index)), shape=(M, N))
        combined_item_vectors = np.zeros((M, N + (self.feature_matrix.shape[1]-1)))

        for item_id in item_list:
            item_index = self.item_to_index[item_id]
            rating_vector = self.rating_matrix[item_index].toarray().flatten()
            feature_vector = self.feature_matrix[self.feature_matrix[self.itemField] == item_id].values[:, 1:].flatten() 
            combined_vector = np.concatenate((rating_vector, feature_vector))
            combined_item_vectors[item_index, :len(combined_vector)] = combined_vector
        self.combined_matrix = combined_item_vectors

    def rated_by_user(self, user):
        #Helper function to get items already rated by user
        user_items = self.rating_matrix.getcol(self.user_to_index[user]).A
        rated_items = list(zip(np.where(user_items > 0)[0],user_items[user_items> 0]))
        self.rated_items = sorted(rated_items, key=lambda x: x[1], reverse=True)
        self.avg_user = np.mean([item[1] for item in self.rated_items])

    def fit(self, metric = None, k = None):
        if metric is None:
            metric = self.primary_distance
        if k is None:
            k = self.k
        self.KNN = NearestNeighbors(n_neighbors=k, algorithm='brute',metric= metric)
        self.KNN.fit(self.combined_matrix)

    def find_similar_items(self, user):
        self.rated_by_user(user)
        self.favorite_indices = [item[0] for item in self.rated_items]
        unseen_idx = []

        for favorite_index in self.favorite_indices:
            item_vector = self.combined_matrix[favorite_index].reshape(1, -1)
            distances, indices = self.KNN.kneighbors(item_vector, n_neighbors=self.n_neighbors)
            combined_list = list(zip(indices[0].tolist(), distances[0].tolist()))[1:]
            filtered_list = [(index, distance) for index, distance in combined_list if index not in self.favorite_indices]
            for index, distance in filtered_list:
                if index not in [item[0] for item in unseen_idx]:
                   unseen_idx.append((index, distance))
            if len(unseen_idx) >= self.n_neighbors:
                break
        
        self.similar_items = unseen_idx
        return self.similar_items
    
    def ratings_similar_items(self,user, n=100):
        distance_calc = Distance()      
        result_list = []                

        for index, _ in self.similar_items: 
            input_vector = self.combined_matrix[index]
            distances = []
            for idx, rating in self.rated_items[:n]:
                target_vector = self.combined_matrix[idx]
                distance = distance_calc.calculate(vector1 = input_vector, 
                                                   vector2 = target_vector, 
                                                   metric=self.prediction_metric)
                distances.append((idx, distance, rating))
            distances = sorted(distances, key=lambda x: x[1])[:10]
            result_list.append((index, distances))

        self.ratings_neighbors = result_list
        return result_list

    def predict_ratings(self):
        '''This function will go through the ratings of the user for the neighbours of the N recommended items.
        Next, there two ways of predicting the ratings for the top N unseen items regression or classification
        (Nikolakopoulos et al. 2021). (more info in demo file)'''

        # 1. Regression prediction method
        if self.predict_type == 'regression':
            weighted_averages = [] 
            for idx, idx_dist_rating in self.ratings_neighbors:
                weighted_sum = sum_inverse_distances = 0
                for idx_nb, distance, rating in idx_dist_rating:
                    if distance != 0:
                        inverse_distance = 1 / distance
                        weighted_sum += inverse_distance * rating
                        sum_inverse_distances += inverse_distance
                if sum_inverse_distances != 0:
                    weighted_avg = weighted_sum / sum_inverse_distances
                else:
                    weighted_avg = self.avg_user
                weighted_averages.append((self.target_user, idx, weighted_avg))
            self.recommendations_predictions = sorted(weighted_averages, key=lambda x: x[1], reverse=True)
        

        # 2. Classification Prediction Method
        elif self.predict_type == 'classification':
            sorted_voters = []

            for item in self.ratings_neighbors:
                idx, entries = item
                entries.sort(key=lambda x: x[1])
                top_vote = entries[0]
                sorted_voters.append((idx, top_vote))
            
            idx_and_predictions = []
            for item in sorted_voters:
                idx_and_predictions.append((self.target_user, item[0], item[1][2]))
        
            self.recommendations_predictions = sorted(idx_and_predictions, key=lambda x: x[2], reverse=True)
        
        else:
            print('Please select valid prediction type (regression or classification)')

    
    def recommend(self, userlist, prints=True):
        n_neighbors = self.n_neighbors
        recommendations = []
        for user in userlist:
            self.target_user = user
            self.find_similar_items(user)
            self.ratings_similar_items(user)
            self.predict_ratings()
            recommendations.append([(userid, self.index_to_item[idx], rating) for userid, idx, rating in self.recommendations_predictions][:n_neighbors]) #convert back to ID's
        recommendations = [sorted(sublist, key=lambda x: x[2], reverse=True) for sublist in recommendations]
        if prints == True:
            print(recommendations)
        else:
            return recommendations