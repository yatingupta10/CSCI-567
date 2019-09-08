from __future__ import division, print_function

from typing import List

import numpy as np
import scipy

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:

    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function

    #TODO: save features and lable to self
    def train(self, features: List[List[float]], labels: List[int]):
        # features: List[List[float]] a list of points
        # labels: List[int] labels of features
        self.features = features
        self.labels = labels

    #TODO: predict labels of a list of points
    def predict(self, features: List[List[float]]) -> List[int]:
        # features: List[List[float]] a list of points
        # return: List[int] a list of predicted labels
        import collections
        from collections import Counter
        
        final_votes=[]
        for a,c in zip(features,self.labels):
            list1 = []
            list2 = []
            for g,h in zip(self.features,self.labels):
                list1.append(g)
                list2.append(h)
            distances=[]
            for item,i in zip(list1,list2):
                dist = self.distance_function(a, item)
                distances.append([dist, i])
            votes = [i[1] for i in sorted(distances)[:self.k]]
            vote_result = Counter(votes).most_common(1)[0][0]
            final_votes.append(vote_result)
        #print(len(self.features))
        return final_votes

    #TODO: find KNN of one point
    #def get_k_neighbors(self, point: List[float]) -> List[int]:
        # point: List[float] one example
        # return: List[int] labels of K nearest neighbor
        #raise NotImplementedError


if __name__ == '__main__':
    print(np.__version__)
    print(scipy.__version__)
