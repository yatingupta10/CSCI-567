import numpy as np
import utils as Util
from typing import List


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        self.feature_dim = len(features[0])
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
            # print ("feature: ", feature)
            # print ("pred: ", pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    #TODO: try to split current node
    def all_attrib_values(self,index):
        values = set()
        for feature in self.features:
            values.add(feature[index])
        return len(values)

    def make_br(self, attribute_index):
        attribute_to_classes = {}
        for feature_index in range(0, len(self.features)):
            attribute_val = self.features[feature_index][attribute_index]
            classes = []
            if attribute_val in attribute_to_classes:
                classes = attribute_to_classes[attribute_val]
            classes.append(self.labels[feature_index])
            attribute_to_classes[attribute_val] = classes  
        branches = []
        for key in attribute_to_classes:
            classes = attribute_to_classes[key]
            unique, counts = np.unique(classes, return_counts=True)
            branches.append(counts)
        return branches

    def split(self):
        if len(self.features)==0:
            self.splittable = False
        
        if len(self.labels) == 0:
            self.splittable = False
        
        if len(self.features[0])==0:
            self.splittable = False
        
        # if not splittable, return majority label
        if self.splittable==False:
            return

        unique, counts = np.unique(self.labels, return_counts=True)
        current_entropy = Util.get_entropy(counts) 

        max_info = None
        for index in range(len(self.features[0])): 
            branches = self.make_br(index)
            info = Util.Information_Gain(current_entropy, branches)
            if max_info is None or info > max_info:
                max_info = info
                self.dim_split = index
            elif info == max_info:
                current_index_values = self.all_attrib_values(index)
                best_index_values = self.all_attrib_values(self.dim_split)
                if current_index_values > best_index_values:
                    self.dim_split = index
                elif current_index_values == best_index_values:
                    self.dim_split = min(self.dim_split, index)

        attribute_val_dict = {}
        for index in range(0, len(self.features)):
            attribute_val = self.features[index][self.dim_split]
            features_and_labels = [[],[]]
            if attribute_val in attribute_val_dict:
                features_and_labels = attribute_val_dict[attribute_val] 
            feature_to_del = self.features[index]
            feature_to_del = np.delete(feature_to_del, [self.dim_split])
            features_and_labels[0].append(feature_to_del)
            features_and_labels[1].append(self.labels[index])
            attribute_val_dict[attribute_val] = features_and_labels
        
        self.feature_uniq_split = list(attribute_val_dict.keys())
        self.feature_uniq_split=sorted(self.feature_uniq_split, key = lambda e: ({int:1, float:1, str:0}.get(type(e), 0), e))
        
        for key in self.feature_uniq_split:
            features_and_labels = attribute_val_dict[key]
            childn = TreeNode(
                    features_and_labels[0],
                    features_and_labels[1],
                    np.unique(features_and_labels[1]))  
            if childn.splittable:
                childn.split()
            self.children.append(childn)
         
    def predict(self, feature: List[int]) -> int:
        if self.splittable:
            if feature[self.dim_split] not in self.feature_uniq_split:
                return self.cls_max
            childi = self.feature_uniq_split.index(feature[self.dim_split])
            feature = np.delete(feature,[self.dim_split])
            return self.children[childi].predict(feature)
        else:
            return self.cls_max