import numpy as np
from typing import List
from hw1_knn import KNN
import math

#a = 0
# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    sum_ent = 0.0
    total_ex = 0.0
    for i in range(len(branches)):
        total_ex += float(sum(branches[i]))
    for i in range(len(branches)):
        g = 0.0
        sum_br = float(sum(branches[i]))
        for j in range(len(branches[i])):
            if branches[i][j] == 0:
                continue
            else:
                g += (branches[i][j]/sum_br)*np.log2(float(branches[i][j])/sum_br)
        sum_ent -= (sum_br/total_ex)*g
    return S - sum_ent

def get_entropy(branch):
    total = np.sum(branch)
    if total == 0:
        return 0
    answer = 0
    for class_count in branch:
        probability = class_count / total
        answer += probability * math.log(probability, 2)  
    return answer * -1


# TODO: implement reduced error prunning function, pruning your tree on this function
def acc_score(actual, predictions):
    good = 0
    for i in range(len(actual)):
        if actual[i] == predictions[i]:
            good += 1
    return good/(len(actual))

def prune(node, decisionTree, X_test, y_test, best_score):
    if node.splittable == False:
        return best_score
   
    for i in range(len(node.children)):
        best_score = prune(node.children[i],decisionTree, X_test, y_test, best_score)
    
    node.splittable = False
    pred = decisionTree.predict(X_test)
    score = acc_score(y_test, pred)
    if score > best_score:
        node.children = []
        best_score = score
    else:
        node.splittable = True
    return best_score

def reduced_error_prunning(decisionTree, X_test, y_test):
    best_f_score = acc_score(y_test, decisionTree.predict(X_test))
    prune(decisionTree.root_node, decisionTree, X_test, y_test, best_f_score)


# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')


#TODO: implement F1 score
def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    assert len(real_labels) == len(predicted_labels)
    tp = sum([x == 1 and y == 1 for x, y in zip(real_labels, predicted_labels)])
    fp = sum([x == 0 and y == 1 for x, y in zip(real_labels, predicted_labels)])
    fn = sum([x == 1 and y == 0 for x, y in zip(real_labels, predicted_labels)])
    if 2 * tp + fp + fn == 0:
        return 0
    f1 = 2 * tp / float(2 * tp + fp + fn)
    return f1

#TODO:
def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    dist = [(x - y)**2 for x, y in zip(point1, point2)]
    dist = np.sqrt(sum(dist))
    return dist


#TODO:
def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    dist = [(x*y) for x, y in zip(point1, point2)]
    dist = sum(dist)
    return dist


#TODO:
def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    dist = [(x - y)**2 for x, y in zip(point1, point2)]
    dist = sum(dist)
    dist = -np.exp(-0.5 * dist)
    return dist


#TODO:
def cosine_sim_distance(point1: List[float], point2: List[float]) -> float:
    return 1.0 - (np.dot(point1, point2) / (np.linalg.norm(point1)*np.linalg.norm(point2)))


# TODO: select an instance of KNN with the best f1 score on validation dataset
def model_selection_without_normalization(distance_funcs, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    best_f1_score, best_k = -1, 0
    for name, func in distance_funcs.items():
        k_lim = len(Xtrain) - 1
        for k in range(1, min(31, k_lim), 2):
            model = KNN(k=k, distance_function=func)
            model.train(Xtrain, ytrain)
            valid_f1_score = f1_score(yval, model.predict(Xval))
            if valid_f1_score > best_f1_score:
                best_f1_score, best_k = valid_f1_score, k
                model1 = model
                func1 = name
    return model1, best_k, func1


# TODO: select an instance of KNN with the best f1 score on validation dataset, with normalized data
def model_selection_with_transformation(distance_funcs, scaling_classes, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # scaling_classes: diction of scalers
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    # return best_scaler: best function choosed for best_model
    best_f1_score, best_k = 0, -1
    for scaling_name, scaling_class in scaling_classes.items():
        for name, func in distance_funcs.items():
            scaler = scaling_class()
            train_features_scaled = scaler(Xtrain)
            valid_features_scaled = scaler(Xval)
            k_lim = len(Xtrain) - 1
            for k in range(1, min(31, k_lim), 2):
                model = KNN(k=k, distance_function=func)
                model.train(train_features_scaled, ytrain)
                valid_f1_score = f1_score(yval, model.predict(valid_features_scaled))
                if valid_f1_score > best_f1_score:
                    best_f1_score, best_k = valid_f1_score, k
                    model1 = model
                    func1 = name
                    scaler1 = scaling_name
    return model1, best_k, func1, scaler1                   



class NormalizationScaler:
    def __init__(self):
        pass

    #TODO: normalize data
    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        normalized = []
        for sample in features:
            if all(x == 0 for x in sample):
                normalized.append(sample)
            else:
                denomin = float(np.sqrt(inner_product_distance(sample, sample)))
                sample_normalized = [x / denomin for x in sample]
                normalized.append(sample_normalized)
        return normalized


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]
    """
    def __init__(self):
        self.count=0
        self.maxi=[]
        self.mini=[]
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        j = []
        a = len(features[0])
        b = len(features)
        data = np.array(features).T
        if self.count == 0:
            for i in range(len(data)):
                self.maxi.append(np.max(data[i]))
                self.mini.append(np.min(data[i]))
            self.count = self.count + 1
        for i in range(b):
            l = []
            for k in range(a):
                c = (features[i][k] - self.mini[k])/(self.maxi[k] - self.mini[k])
                l.append(c)
            j.append(l)
        t = np.matrix(j)
        f = t.tolist()
        return f