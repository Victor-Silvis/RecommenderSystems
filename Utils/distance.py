import numpy as np

'''
This code is used as helper function for calculating the distance between two vectors.
In the context of the KNN recommender system. this code is used to calculate distance to items
the user has already seen that are similar to the unseen item we are going to recommend. Based
on that the predictions are made.

args
metric  :   Choice of metric, manhattan by default
'''


class Distance:

    def __init__(self, metric='manhattan'):
        self.metric = metric

    def calculate(self, vector1, vector2, metric):
        self.vector1 = vector1
        self.vector2 = vector2
        
        if metric is not None:
            self.metric = metric

        if self.metric == 'cosine':
            dist = self.cosine_similarity()
        elif self.metric == 'manhattan':
            dist = self.manhattan_distance()
        elif self.metric == 'euclidean':
            dist = self.euclidean_distance()
        else:
            print('Not a valid metric')
        return dist

    def manhattan_distance(self):
        distance = np.sum(np.abs(self.vector1 - self.vector2))
        return distance
    
    def cosine_similarity(self):
        dot_product = np.dot(self.vector1, self.vector2)
        norm_vector1 = np.linalg.norm(self.vector1)
        norm_vector2 = np.linalg.norm(self.vector2)
        similarity = dot_product / (norm_vector1 * norm_vector2)
        return similarity
    
    def euclidean_distance(self):
        squared_diff = np.sum((self.vector1 - self.vector2) ** 2)
        distance = np.sqrt(squared_diff)
        return distance
