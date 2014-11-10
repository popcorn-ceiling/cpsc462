#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""hw5.py:  Data mining assignment #4: Ensemble methods."""

__author__ = "Dan Collins and Miranda Myers"
import copy
import csv
import random
import operator
import numpy
import numpy.ma as ma
from math import log
from tabulate import tabulate

class DecisionTreeClassifier:

    def __init__(self, fileName, classIndex):
        """Constructor for DTClassifier class."""
        self.table = self.read_csv(fileName)
        self.attrNames = self.table.pop(0)
        self.classIndex = classIndex
        self.uniqueClasses = []
        for row in self.table:
            if row[self.classIndex] not in self.uniqueClasses:
                self.uniqueClasses.append(row[self.classIndex])
        self.uniqueClasses.sort()
        self.decisionTree = None

    def read_csv(self, fileName):
        """Reads in a csv file and returns a table as a list of lists (rows)."""
        theFile = open(fileName)
        theReader = csv.reader(theFile, dialect='excel')
        table = []
        for row in theReader:
            if len(row) > 0:
                table.append(row)
        return table
        
    def get_column_as_floats(self, table, index):
        """Returns all non-null values in table for given index as floats."""
        vals = []
        for row in table:
            if row[index] != "NA":
                vals.append(float(row[index]))
        return vals 
        
    def get_column_as_strings(self, table, index):
        """Returns all non-null values in table for given index as strings."""
        vals = []
        for row in table:
            if row[index] != "NA":
                vals.append(str(row[index]))
        return vals
    
    def partition_classes(self, classIndex, className, table):
        """Given a class name and index, return a table of instances \
           that contain that class."""
        classPartition = []
        for row in table:

                try:
                    if float(row[classIndex]) == float(className):
                        classPartition.append(row)
                except ValueError:
                    # compare as strings 
                    if row[classIndex] == className:
                        classPartition.append(row)

        return classPartition

    def k_cross_fold_partition(self, table, k, classIndex, curBin):
        """Partition a data set into train and test by splitting into K folds (bins)."""
        # randomize 
        randomized = table # table passed is already a copy
        n = len(table)
       
        for i in range(n):
            # pick an index to swap
            j = random.randint(0, n-1) # random int [0,n-1] inclusive
            randomized[i], randomized[j] = randomized[j], randomized[i]

        # get classes
        classNames = []
        for row in randomized:
            if row[classIndex] not in classNames:
                classNames.append(row[classIndex])
        
        # partition dataTitle - each subset contains rows with a unique class
        dataPartition = []
        for i in range(len(classNames)):
            dataPartition.append(\
                self.partition_classes(classIndex, classNames[i], randomized))

        # distribute paritions roughly equally
        kPartitions = [[] for _ in range(k)]
        for partition in dataPartition:
            for i in range(len(partition)):
                kPartitions[i%k].append(partition[i])
        
        # return training and test set
        testSet = kPartitions[curBin]
        trainingSet = []
        for i in range(k):
            if i != curBin:
                trainingSet += kPartitions[i]
        return trainingSet, testSet  
        
    def in_same_class(self, instances, classIndex):
        """Returns true if all instances have same label."""
        # Get first label
        testLabel = instances[0][self.classIndex]
        
        # Test whether all instances have the same label
        for instance in instances:
            if instance[self.classIndex] != testLabel:
                return False
                
        return True
        
    def partition_stats(self, instances):
        """Return a dictionary of stats: {(classValue, tot1), (classValue, tot2), ...}."""
        statDictionary = {}
        for instance in instances:
            if instance[self.classIndex] not in statDictionary:
                statDictionary.update({instance[self.classIndex] : 1})
            else:
                statDictionary[instance[self.classIndex]] += 1
            instances.pop()
            
        return statDictionary
        
    def partition_instances(self, instances, attIndex):
        """Partition list: {attval1:part1, attval2:part2}."""
        # partitions looks like
        # [ [value_of_parition(i.e 1), [[ ...inst...],
        #                               [ ...inst...],
        #                               ...
        #                              ]
        #   ]
        # ]

        values = []
        for i in range(len(instances)):
            if instances[i][attIndex] not in values:
                values.append(instances[i][attIndex])
        
        subpartition = [[] for _ in range(len(values))]
        for i in range(len(instances)):
            index = values.index(instances[i][attIndex])
            subpartition[index].append(instances[i])

        partitions = []
        for i in range(len(values)):
            partitions.append([values[i], subpartition[i]])
        
        return partitions
    
    def calculate_pi(self, classIndex, instances):
        """Returns the probability of each value occurring in a column."""
        column = self.get_column_as_strings(instances, classIndex) 
        sortedColumn = sorted(column)
        totalLabels = len(sortedColumn)
        
        labels, probabilities = [], []
        for label in sortedColumn:
            if label not in labels:
                labels.append(label)
                probabilities.append(1)
            else:
                probabilities[-1] += 1
        
        for i in range(len(probabilities)):
            probabilities[i] /= float(totalLabels)
        
        return labels, probabilities
        
    def calculate_entropy(self, instances):
        """Calculates shannon entropy on a set of instances."""
        # Get all pi values
        labels, probabilities = self.calculate_pi(self.classIndex, instances)
        
        # Iterate through the class labels of the given instances to calculate entropy
        E = 0
        for label in labels:
            # pi is the proportion of instances with given label
            pi = probabilities[labels.index(label)]
            E -= -(pi * log(pi, 2))
        
        return E     
    
    def calculate_Enew(self, instances, attIndex): 
        """Calculate Enew for a single attribute."""
        # Partition instances on attribute 
        partitions = self.partition_instances(instances, attIndex)        
                
        # Calculate Enew for the instances partitioned on the given attribute
        Enew = 0
        for partition in partitions:
            EDj = self.calculate_entropy(partition[1])
            Dj = len(partition[1])
            D = len(instances)
            Enew += (Dj / D) * EDj
        return Enew
        
    def find_smallest_Enew(self, instances, attIndices):
        """Find which attribute partition gives the smallest attribute value - the
            smallest amount of information needed to classify an instance."""
        EnewList = []
        for attIndex in attIndices:
            Enew = self.calculate_Enew(instances, attIndex)
            EnewList.append(Enew)
        
        smallestEnew = min(EnewList)
        attIndex = attIndices[EnewList.index(smallestEnew)]
        return attIndex

    def select_attribute(self, instances, attIndices):
        """Returns attribute index to partition on using chosen selection method."""
        attIndex = self.find_smallest_Enew(instances, attIndices) 
        return attIndex
        
    def resolve_clash(self, statDictionary):
        """."""
        #TODO what happens if it's 50/50? Selecting first item for now
        values = list(statDictionary.values())
        keys = list(statDictionary.keys())
        return keys[values.index(max(values))]
            
    def tdidt(self, instances, attIndices, f=-1):
        """Returns tree object.
           Uses Top Down Induction of Decision Trees recursive algorithm.
           Algorithm:
            - At each step, pick an attribute ("attribute selection")
            - Partition data by attribute values ... pairwise disjoint partitions
            - Repeat until (base cases):
                 1. Partition has only class labels that are the same ... no clashes
                 2. No more attributes to partition ... there may be clashes
                 3. No more instances to partition ... backtrack, create single leaf node
        """        
        # Repeat until base case(s)
        # No more instances to partition
        if len(instances) == 0:
            return
        # No more attributes to partition
        if len(attIndices) == 0:
            stats = self.partition_stats(instances)
            label = self.resolve_clash(stats)
            return ['label', label]
        # Only class labels that are the same
        elif self.in_same_class(instances, self.classIndex):
            label = instances[0][self.classIndex]
            return ['label', label]

        # At each step select an attribute and partition data      
        if f != -1:
            # select new attribute from subset of size f from remaining attributes
            attIndicesRand = self.select_random_attributes(f, attIndices)
            attr = self.select_attribute(instances, attIndicesRand)
        else:
            # select new attribute from remaining attributes
            attr = self.select_attribute(instances, attIndices)

        partitions = self.partition_instances(instances, attr)

        node = ['attribute', self.attrNames[attr], []]
        attrRemaining = [item for item in list(attIndices) if item != attr]
        
        for item in partitions:
            subtree = self.tdidt(item[1], attrRemaining, f)
            node[2].append(['value', item[0], subtree])
        
        return node

    def dt_get_subtree_classes(self, st, subTreeClasses):
        """Gets classes on all subtree paths and returns them in a list of dictionaries.
           [ {class0:count0}, {class1:count1}, ..., {classn:countn} ]."""
        nodeType = st[0]
        nodeVal  = st[1]
        if (nodeType == 'label'):
            if nodeVal not in subTreeClasses:
                subTreeClasses.update({nodeVal : 1})
            else:
                subTreeClasses[nodeVal] += 1
        else:
            for path in st[2]:
                subTreeClasses = self.dt_get_subtree_classes(path[2], subTreeClasses)
        return subTreeClasses
   
    def dt_print(self, dt, level=1):
        """Debug print function for trees."""
        if len(dt) > 0:
            if len(dt) > 2:
                print '|' + (level * '---'), dt[0],':',dt[1]
                for item in dt[2]:
                    print '|' +  ((level + 1) * '---'), item[0], ':', item[1]
                    self.dt_print(item[2], level + 2)
            else:
                print '|' + (level * '---'), dt[0],':',dt[1]
                return

    def dt_classify(self, dt, instance, att='ROOT'):
        """Classifies an instance using a decision tree passed to it."""
        nodeType = dt[0]
        nodeVal  = dt[1]
        if (nodeType == 'attribute'):
            # find the index of attribute based on list of column titles
            attIndex = self.attrNames.index(nodeVal)
        elif (nodeType == 'value'):
            # need to know the attribute this value node belongs to
            # this was passed from previous recursion level
            attIndex = self.attrNames.index(att)
        else: 
            # label node, we're done!
            return nodeVal

        instVal = instance[attIndex]           
        nodeSubTree = dt[2]
        label = 'NOCLASS'
        # search the child nodes for matches with our instance
        for child in nodeSubTree:
            childVal = child[1]
            if (instVal == childVal):
                label = self.dt_classify(child[2], instance, nodeVal)

        # we couldn't find a path; majority vote on all possible subtrees
        if (label == 'NOCLASS'):
            for child in nodeSubTree:
                stClasses = self.dt_get_subtree_classes(child[2], {})
            values = list(stClasses.values())
            keys = list(stClasses.keys())
            label = keys[values.index(max(values))]
        return label

    def build_rand_forest_ens(self, title, remainder, attIndices, f, m, n):
        """Given a remainder set, builds a random forest of N trees and creates
           and ensemble by selecting the M most accurate. At each node, a 
           random attribute of F of the remaining attributes is selected."""

        # build N trees
        forest, predAccs, cfMatrices = [], [], []
        for _ in range(n):
            # partition
            trainSet, valSet = self.bootstrap(remainder)
            # build individual tree
            forest.append(self.tdidt(trainSet, attIndices, f))
            # classify test set using each tree and calculate accuracy
            predAcc, cfMatrix = self.calculate_accuracy(forest[-1], valSet)
            predAccs.append(predAcc)  
            cfMatrices.append(cfMatrix)
          
        # Select the M most accurate trees with associated confusion matrices
        topTrees = self.select_most_accurate(predAccs, forest, cfMatrices, m)
        return topTrees

    def calculate_accuracy(self, tree, valSet):
        """Return the predictive accuracy for a given tree and test set."""
        # For each instance find the predicted label and actual label
        labels, actual = [], []
        for instance in valSet:
            actual.append(instance[self.classIndex])
            labels.append(self.dt_classify(tree, instance))
         
        # create confusion matrix for track record voting
        cfMatrix = self.create_confusion_matrix('forest member', labels, actual)

        # Calculate number of correct classifications  
        correct = 0 
        for i in range(len(actual)):
            if actual[i] == labels[i]:
                correct += 1
         
        # Predictive accuracy = (TP + TN) / all      
        predAcc = correct / (len(actual) * 1.0)
        return predAcc, cfMatrix
        
    def select_most_accurate(self, predAccs, forest, cfMatrices, M):
        """Given a forest and its corresponding predictive accuracies,
           return a list of the trees with the highest accuracy.  Select the
           M most accurate of the N decision trees"""
        
        # Convert list to numpy arrays to make use of masked array
        predAccs = numpy.array(predAccs)  
        topTrees = []
        
        # FIXME - clean up this fxn
        while len(topTrees) < M:
            # Find the highest predictive accuracy
            maxAccuracy = max(predAccs)
            
            while len(topTrees) < M:
                # Append the tree(s) with the max accuracy to topTrees
                [topTrees.append([forest[i], cfMatrices[i]]) \
                for i, j in enumerate(predAccs) \
                if j == maxAccuracy and len(topTrees) < M]

            # Mask the maximum value
            predAccs = ma.masked_equal(predAccs, maxAccuracy)
        return topTrees

    def create_confusion_matrix(self, dataTitle, classLabels, actualLabels):
        """Creates confusion matrix for a given set of classified vs actual labels."""
        # create empty confusion matrix
        cfMatrix = [[0 for i in range(2 + len(self.uniqueClasses))] 
                              for x in range(2 + len(self.uniqueClasses))]
        # warning - gross abuse of casting to follow
        # add labels 
        cfMatrix[0][0]  = dataTitle
        cfMatrix[0][-1] = 'Total'
        cfMatrix[-1][0] = 'Total'
        for i in range(1, len(cfMatrix[0]) - 1):
            cfMatrix[0][i] = str(self.uniqueClasses[i - 1])
            cfMatrix[i][0] = str(self.uniqueClasses[i - 1])
        
        # place values
        for i in range(len(classLabels)):
            eIndex = cfMatrix[0].index(str(classLabels[i]))
            aIndex = cfMatrix[0].index(str(actualLabels[i]))
            cfMatrix[aIndex][eIndex] += 1

        # tally totals
        for row in range(1, len(cfMatrix[0]) - 1):
            for col in range(1, len(cfMatrix[0]) - 1):
                cfMatrix[row][-1] += cfMatrix[row][col]
                cfMatrix[-1][col] += cfMatrix[row][col]
                # convert to string because reasons (thanks tabulate)
                cfMatrix[row][col] = str(cfMatrix[row][col])
            cfMatrix[row][-1] = str(cfMatrix[row][-1])
            cfMatrix[-1][-1] += int(cfMatrix[row][-1])
        
        # convert last row to strings too...  
        for col in range(1, len(cfMatrix[0])):
            cfMatrix[-1][col] = str(cfMatrix[-1][col])
        
        # calculate recognition
        cfMatrix[0].append('Recognition Rate')
        for row in range(1, len(cfMatrix[0]) - 2):
            hits = float(cfMatrix[row][row])
            total = float(cfMatrix[row][-1])
            if total == 0:
                recognition = 0
            else:
                recognition = round(hits/total, 2)
            cfMatrix[row].append(str(recognition))
        cfMatrix[-1].append('NA')

        return cfMatrix
        
    def select_random_attributes(self, F, attributes):
        """Randomly select F of the remaining attributes as candidates to partition on."""
        if len(attributes) <= F:
            return attributes
        random.shuffle(attributes)
        return attributes[:F]        

    def bootstrap(self, table):
        """."""
        trainingSet, testSet = [], []
        used = []
        for i in range(len(table)):
            rand = random.randint(0, len(table) - 1)
            used.append(rand)
            trainingSet.append(table[rand])
        for i in range(len(table)):
            if i not in used:
                testSet.append(table[i])
        return trainingSet, testSet
        
    def majority_vote(self, labels):
        """."""
        freqDict = {}
        for l in labels:
            if l not in freqDict:
                freqDict.update({l:1})
            else:
                freqDict[l] += 1
                
        keys = freqDict.keys()
        cts = freqDict.values()
        
        return keys[cts.index(max(cts))]
    
    def track_record_vote(self, cfMatrix, label):
        """Given a confusion matrix and a label, returns a dict of class labels
           and their corresponding split vote percentages."""
        i = cfMatrix[0].index(label)
        voteCol = self.get_column_as_strings(cfMatrix, i)
        labelCol = self.get_column_as_strings(cfMatrix, 0)

        splitVote = {}
        for j in range(1, len(voteCol) - 1):
            splitVote.update( {labelCol[j] : float(voteCol[j])/float(voteCol[-1])} )
    
        return splitVote

    def test_rand_forest_ens(self, title):
        """."""
        attIndices = [i for i in range(0, len(self.table[0]))]
        attIndices.pop(attIndices.index(self.classIndex))
        f, m, n = 5, 5, 15     
        if (m > n):
            print 'ERROR: M must be less than or equal to N'
            exit(-1)

        print '==============================================================='
        print 'STEP 1: Random Forest vs Standard Tree (', title, ')'
        print '==============================================================='
        print
        print 'Random Forest with N =', n, 'M =', m, 'F =', f
        
        # partition data into 2/3 remainder set and 1/3 test set
        k = 3
        remainderSet, testSet =  self.k_cross_fold_partition( \
                           self.table, k, self.classIndex, 0)

        # build forest and select M top trees using track record voting
        # format [ [tree0, cfMatrix0], [tree1, cfMatrix1], ... ]
        topM = self.build_rand_forest_ens(title, remainderSet, attIndices, f, m, n)
        # use same remainder set for std entropy tree on all avail attr
        stdTree = self.build_rand_forest_ens(title, remainderSet, attIndices, -1, 1, 1)
        
        labelsForest, labelsTree, actual = [], [], []
        classPoints = {}
        # test with test set
        for instance in testSet:
            for i in range(len(topM)):
                tree = topM[i][0]
                cfMatrix = topM[i][1]
                classLocal = self.dt_classify(tree, instance)
                # splitVote is a dictionary of classes and their vote values
                splitVote = self.track_record_vote(cfMatrix, classLocal)
                for item in splitVote.keys():
                    if item not in classPoints:
                        classPoints.update(splitVote)
                    else:
                        classPoints[item] += splitVote[item]    
            label = max(classPoints, key=classPoints.get)
            labelsForest.append(label)
            labelsTree.append(self.dt_classify(stdTree[0][0], instance))
            actual.append(instance[self.classIndex])

        # build confusion matrix
        cfMatrixForest = self.create_confusion_matrix(title, labelsForest, actual)
        print tabulate(cfMatrixForest)
        print
        print 'Standard Tree with F = len(attributes)'
        cfMatrixTree = self.create_confusion_matrix(title, labelsTree, actual)
        print tabulate(cfMatrixTree)

def main():
    """Creates objects to parse data file and create trees used for classification."""
    mushroom = DecisionTreeClassifier('agaricus-lepiota.txt', 0)
    mushroom.test_rand_forest_ens('agaricus-lepiota.txt')

    titanic = DecisionTreeClassifier('titanic.txt', 3)
    titanic.test_rand_forest_ens('titanic.txt')

    tictactoe = DecisionTreeClassifier('tic-tac-toe.txt', 9)
    tictactoe.test_rand_forest_ens('tic-tac-toe.txt')

if __name__ == "__main__":
    main()

