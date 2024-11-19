'''
SVD++ Recommender System

This script contains a SVD++ model. This is an extension of the SVD method. While SVD 
focuses solely on decomposing the user-item interaction matrix and making predictions based 
on that decomposition, SVD++ incorporates additional information into the model to enhance its 
accuracy, in this case SVD++ takes into account implicit feedback. Implicit feedback in this code
is the user and item bias. This will be added on top of the factorized matrix.
'''

import numpy as np


class SVD_PP:

    def __init__(self, userField, itemField, valueField):

        ''' 
        args for SVD Recommender:
        userField   : The name of the column that contains user_id's
        itemField   : The name of the column that contains item_id's
        valueField  : the name of the column that contains the values (ratings)
        '''

        self.params = {}
        self.userField = userField
        self.itemField = itemField
        self.valueField = valueField
        self.set_params()


    def set_params(self, k=2, lr=0.003, reg_all=0.5, reg_ui=0.05, reg_bi=0.05, n_epochs=50):
        '''This function sets the initial hyperparameters. The default ones are mostly based
        on literature. However, they get automatically updated after performing hyper
        parameter tuning (see demo)'''

        self.params['k'] = k
        self.params['lr'] = lr
        self.params['reg_all'] = reg_all
        self.params['reg_ui'] = reg_ui
        self.params['reg_bi'] = reg_bi
        self.params['n_epochs'] = n_epochs


    def create_matrix(self, data):

        #Create a utility matrix
        utility_matrix = data.pivot(index=self.userField, 
                                    columns=self.itemField,
                                    values=self.valueField)
        #Create mapping
        N = data[self.userField].nunique()
        M = data[self.itemField].nunique()
        user_list = np.unique(data[self.userField])
        item_list = np.unique(data[self.itemField])
        self.user_to_index = dict(zip(user_list, range(0, N)))
        self.index_to_item = dict(zip(range(0,M), item_list))

        #Create Matrix Operations
        matrix_array = utility_matrix.values #To array
        self.max_value = np.nanmax(matrix_array[(matrix_array != 0) & (~np.isnan(matrix_array))]) #get max value for later
        self.min_value = np.nanmin(matrix_array[(matrix_array != 0) & (~np.isnan(matrix_array))]) #get min value for later
        mask = np.isnan(matrix_array) #Get nan value mask True/False (items user has not explicitly interacted with yet mask)
        masked_arr = np.ma.masked_array(matrix_array, mask) 
        self.predMask = ~mask 
        item_means = np.mean(masked_arr, axis=0) 
        matrix = masked_arr.filled(0) #Replace NaN with 0 (instead of means as done with normal SVD)
        self.item_means_tiled = np.tile(item_means, (matrix.shape[0], 1)) 
        self.matrix = matrix - self.item_means_tiled

#SVD fit
    def sigmoid(self, x): #Sigmoid helper function for user and item biases.
        return 1 / (1 + np.exp(-x))

    def fit(self, k=None, lr=None, reg_all=None, reg_ui=None, reg_bi=None, n_epochs=None, get_params=True):
        
        if get_params == True:
            k = self.params['k']    
            lr = self.params['lr']
            reg_all = self.params['reg_all']
            reg_ui = self.params['reg_ui']
            reg_bi = self.params['reg_bi']
            n_epochs = self.params['n_epochs']

        num_users, num_items = self.matrix.shape # Get dimensions
        U = np.random.normal(scale=1. / k, size=(num_users, k)) # Init user latent factor matrix
        V = np.random.normal(scale=1. / k, size=(num_items, k)) # init item latent factor matrix
        bu = np.zeros(num_users) # Initialize user bias to zero
        bi = np.zeros(num_items) # Initialize item bias to zero
        mu = np.mean(self.matrix[np.where(self.matrix != 0)]) #Calc global mean

        for epoch in range(n_epochs): #Iterate over epochs
            for i in range(num_users): #Iterate over users

                idx_items_rated_by_user = np.where(self.matrix[i, :] > 0)[0] # Select non zero indices
                if len(idx_items_rated_by_user) == 0:
                    continue  # Skip if the user hasn't rated any items
                U_i = U[i, :] # Extract latent factors 
                bu_i = bu[i] # Extract biases
                for j in idx_items_rated_by_user:
                    V_j = V[j, :] # Extract for current item
                    bi_j = bi[j] # Extract for current item
                    prediction = mu + self.sigmoid(bu_i) + self.sigmoid(bi_j) + np.dot(U_i, V_j) #Calc predictions Vectorized
                    eij = self.matrix[i, j] - prediction # Error computation vectorized

                    # Update biases and latent factors
                    bu[i] += lr * (eij - reg_all * self.sigmoid(bu_i) - reg_ui * bu_i) # Vectorized bias update for user
                    bi[j] += lr * (eij - reg_all * self.sigmoid(bi_j) - reg_bi * bi_j) # Vectorized bias update for item
                    U[i, :] += lr * (eij * V_j - reg_all * U_i) # Vectorized latent factor update for user
                    V[j, :] += lr * (eij * U_i - reg_all * V_j) # Vectorized latent factor update for item

        # Reconstruct ratings matrix
        '''Here we recompose the matrix again, but this time we add the item and user
        biases on top the the factorized matrix.
        '''          
        bu_matrix = np.tile(bu, (num_items, 1)).T
        bi_matrix = np.tile(bi, (num_users, 1)) 
        ratings_matrix = mu + self.sigmoid(bu_matrix) + self.sigmoid(bi_matrix) + np.dot(U, V.T) # Vectorized reconstruction of ratings matrix
        self.prediction_ratings = ratings_matrix + self.item_means_tiled # Add means back in for final predictions
        self.prediction_ratings = np.clip(self.prediction_ratings, self.min_value, self.max_value)

  #Recommender   
    def recommend(self, users_list, N=10, values=True):
        predMat = np.ma.masked_where(self.predMask, self.prediction_ratings).filled(fill_value=-999)
        recommendations = [] 

        if values == True:
            for user in users_list:
                try:
                    user_idx = self.user_to_index[user]
                except:
                    raise Exception("Invalid User: ", user)
                top_indeces = predMat[user_idx,:].argsort()[-N:][::-1] 
                recommendations.append([(user, self.index_to_item[index], predMat[user_idx, index]) for index in top_indeces])

        if values == False:
            for user in users_list:
                try:
                    user_idx = self.user_to_index[user]
                except:
                    raise Exception("Invalid User:", user)
                top_indeces = predMat[user_idx,:].argsort()[-N:][::-1]
                recommendations.append([self.index_to_item[index] for index in top_indeces])
        
        return recommendations
    

    #For demo of output, see SVD demo file