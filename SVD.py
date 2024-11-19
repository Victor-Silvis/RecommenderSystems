'''
SVD Recommender System


Singular Value Decomposition. Utilizing matrix factorization to predict what a particular user will rate a item, that
he or she hasn't seen yet. From these predictions, the appropriate items can be recommended to this user. SVD is a well
known and utilized system due to its accuracy and speed. The SVD demo file in this directory will show the effectiveness
of the system (as well as SVD++) on some well known datasets. Futhermore, it will also contain some more information, 
this script is just the code for the system itself.

'''

#Packages
import numpy as np
from scipy.linalg import sqrtm


class SVD:

    def __init__(self, userField, itemField, valueField, k=20):
        
        ''' 
        args for SVD Recommender:
        userField   : The name of the column that contains user_id's
        itemField   : The name of the column that contains item_id's
        valueField  : the name of the column that contains the values (ratings)
        '''
        
        self.userField = userField
        self.itemField = itemField
        self.valueField = valueField
        self.k = 20


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
        matrix_array = utility_matrix.values 
        mask = np.isnan(matrix_array) #Get nan value mask True/False (items user has not interacted with yet)
        masked_arr = np.ma.masked_array(matrix_array, mask)
        self.predMask = ~mask #store the inverse of the mask for later recommendation (for what the user already has watched)
        item_means = np.mean(masked_arr, axis=0) 
        matrix = masked_arr.filled(item_means)
        self.item_means_tiled = np.tile(item_means, (matrix.shape[0], 1))
        self.matrix = matrix - self.item_means_tiled


    def fit_svd(self, k = None):
        
        if k is None:
            k = self.k

        #SVD (M=UΣV^T)
        U, s, V = np.linalg.svd(self.matrix, full_matrices=False)
        s = np.diag(s)
        #next we take only K most significant features
        s = s[0:k,0:k] #Select the top K diagonal elements
        U = U[:,0:k] #Keep the first K columns of U
        V = V[0:k,:] #Keep the first K rows of V
        s_root = sqrtm(s) #Compute the square root of the diagonal matrix s
        Usk = np.dot(U, s_root) # Multiply U by the square root of s
        skV = np.dot(s_root, V) # Multiply the square root of s by V
        UsV = np.dot(Usk, skV) # Compute the reconstructed matrix UsV by multiplying Usk and skV
        self.UsV = UsV + self.item_means_tiled #we add the means back in to get final predicted ratings
      
    def recommend(self, users_list, N=10, values=True):
        predMat = np.ma.masked_where(self.predMask, self.UsV).filled(fill_value=-999)
        recommendations = [] #init list in which recommendations will be stored

        if values == True:
            for user in users_list:
                try:
                    user_idx = self.user_to_index[user]
                except:
                    raise Exception("Invalid User: ", user)
                top_indeces = predMat[user_idx,:].argsort()[-N:][::-1] #access entire row, sort on ratings, take N max, return indeces of columns(items)
                recommendations.append([(user, self.index_to_item[index], predMat[user_idx, index]) for index in top_indeces])
                #Above: Loop over top indeces for this user, retreive the rating for each index and store as tuple, in list

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