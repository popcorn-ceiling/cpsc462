#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""hw4.py:  Data mining assignment #4: Decision Trees.
            Reads in two data sets, auto-mpg and titanic. Builds and classifies
            them using decision trees. Creates confusion matrices to show results."""

__author__ = "Dan Collins and Miranda Myers"

import copy
import csv
import random
import operator
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
    
    def discretize_mpg_doe(self, y): 
        """Classify MPG using the department of energy rating system."""
        y = float(y)          
        if y < 14.0:
            rating = 1
        elif y == 14.0:
            rating = 2
        elif y > 14.0 and y <= 16.0:
            rating = 3
        elif y > 16.0 and y <= 19.0:
            rating = 4
        elif y > 19.0 and y <= 23.0:
            rating = 5
        elif y > 23.0 and y <= 26.0:
            rating = 6
        elif y > 26.0 and y <= 30.0:
            rating = 7
        elif y > 30.0 and y <= 36.0:
            rating = 8
        elif y > 36.0 and y <= 44.0:
            rating = 9
        elif y > 44.0:
            rating = 10
                
        return rating

    def discretize_weight_nhtsa(self, strWeight):
        """Discretize a given weight according to NHTSA vehicle size ranking."""
        weight = float(strWeight)
        if weight < 2000.0:
            categoricalWeight = '1'
        elif weight >= 2000.0 and weight < 2500.0:
            categoricalWeight = '2'
        elif weight >= 2500.0 and weight < 3000.0:
            categoricalWeight = '3'
        elif weight >= 3000.0 and weight < 3500.0:
            categoricalWeight = '4'
        elif weight >= 3500.0:
            categoricalWeight = '5'
        else:
            print 'error in discretize_weight'
            exit(-1)
    
        return categoricalWeight

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
        """Partition a dataTitle into training and test by splitting into K folds (bins)."""
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
        '''Returns true if all instances have same label.'''
        # Get first label
        testLabel = instances[0][self.classIndex]
        
        # Test whether all instances have the same label
        for instance in instances:
            if instance[self.classIndex] != testLabel:
                return False
                
        return True
        
    def partition_stats(self, instances):
		'''Return a dictionary of stats: {(classValue, tot1), (classValue, tot2), ...}.'''
		statDictionary = {}
		for instance in instances:
		    if instance[self.classIndex] not in statDictionary:
		        statDictionary.update({instance[self.classIndex] : 1})
		    else:
		        statDictionary[instance[self.classIndex]] += 1
		    instances.pop()
		    
		return statDictionary
        
    def partition_instances(self, instances, attIndex):
        '''Partition list: {attval1:part1, attval2:part2}.'''
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
        '''FIXME.'''
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
        '''Calculate Enew for a single attribute.'''
        # Partition instances on attribute 
        partitions = self.partition_instances(instances, attIndex)        
                
        # Calculate Enew for the instances partitioned on the given attribute
        Enew = 0
        for partition in partitions:
            EDj = self.calculate_entropy(partition[1])
            Dj = len(partition[1])
            D = len(instances)
            Enew += (Dj / D) * EDj      # TODO is this being calculated right????
        return Enew
        
    def calculate_Enew_split_pt(self, instances, attIndex, partitions):
        # Calculate Enew for the instances partitioned on the given attribute
        Enew = 0
        for partition in partitions:
            EDj = self.calculate_entropy(partition[1])
            Dj = len(partition[1])
            D = len(instances)
            Enew += (Dj / D) * EDj
        return Enew
       
    def find_smallest_Enew(self, instances, attIndices):
        '''Find which attribute partition gives the smallest attribute value - the
            smallest amount of information needed to classify an instance.'''
        EnewList = []
        for attIndex in attIndices:
            Enew = self.calculate_Enew(instances, attIndex)
            EnewList.append(Enew)
        
        smallestEnew = min(EnewList)
        attIndex = attIndices[EnewList.index(smallestEnew)]
        return attIndex

    def select_split_point(self, instances, attIndex):
        '''Given a set of instances and an attribute index, find the split point to 
            partition the data that gives the lowest Enew.'''
        EnewList = []
        splitPointList = []

        # sort the values in ascending order [v1, v2, . . . , vk]
        # For each split point v in v1 through vk-1 calculate Enew
        values = []
        for row in instances:
            if row[attIndex] not in values:
                values.append(row[attIndex])
        values.sort()
        
        if (len(values) < 2):
            for i in range(1, len(values) - 1):
                splitPoint = values[i]           
                partitions = self.partition_on_split_pt(instances, attIndex, splitPoint)
                
                Enew = self.calculate_Enew_split_pt(instances, attIndex, partitions)
                EnewList.append(Enew)
        else:
            # if we only have 2 items, split is done basically
            splitPoint = float(values[0]) + 0.5
            return str(splitPoint)
        
        smallestEnew = min(EnewList)
        return values[EnewList.index[smallestEnew]]
          
    def partition_on_split_pt(self, instances, attIndex, splitPoint):
        '''Given a list of instances, the index of the attribute we are splitting on,
            and the chosen split point, create the partitions.'''
        # partitions looks like
        # [ [lte:splitPoint,            [[ ...inst...],
        #                               [ ...inst...],
        #                                       ...]]
        #                                           ]
        #   [gte:splitPoint,            [[...inst...],
        #                                [...inst...]   
        #                                   .......]]
        # ]

        leftPoint = 'lte:' + str(splitPoint)
        rightPoint = 'gt:' + str(splitPoint)
        values = [leftPoint, rightPoint]

        subpartition = [[], []]
        for instance in instances:
            if instance[attIndex] <= splitPoint:
                subpartition[0].append(instance)
            else:
                subpartition[1].append(instance)

        partitions = []
        for i in range(len(values)):
            partitions.append([values[i], subpartition[i]])
    
        return partitions
        
                
    def select_attribute(self, instances, attIndices, selectionType):
        '''Returns attribute index to partition on using chosen selection method.'''
        #if selectionType == 'continuous':
        attIndex = self.find_smallest_Enew(instances, attIndices) #, selectionType)
        return attIndex
        
    def resolve_clash(self, statDictionary):
        '''.'''
        #TODO what happens if it's 50/50?
        values = list(statDictionary.values())
        keys = list(statDictionary.keys())
        return keys[values.index(max(values))]
            
    def tdidt(self, instances, attIndices, selectType):
        '''Returns tree object.
           Uses Top Down Induction of Decision Trees recursive algorithm.
           Algorithm:
            - At each step, pick an attribute ("attribute selection")
            - Partition data by attribute values ... pairwise disjoint partitions
            - Repeat until (base cases):
                 1. Partition has only class labels that are the same ... no clashes
                 2. No more attributes to partition ... there may be clashes
                 3. No more instances to partition ... backtrack, create single leaf node
        '''        
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

        # FIXME: if we are partitioning on an attribute, even for continuous we should do the normal partition, I think
                #I'm pretty sure that only when it's continuous, and we are partitioning the values within an att node,
                    # is when we use the split point method.  So we have to figure out some way to tell which type of node
                        # we are on and which type of data (categorial/cont) it is, then partition accordingly
                            
                            #.....right?
        
        # At each step select an attribute and partition data      
        attr = self.select_attribute(instances, attIndices, selectType)
        if selectType == 'categorical':
            partitions = self.partition_instances(instances, attr)
        else:
            splitPoint = self.select_split_point(instances, attr)
            partitions = self.partition_on_split_pt(instances, attr, splitPoint)

        node = ['attribute', self.attrNames[attr], []]
        attrRemaining = [item for item in list(attIndices) if item != attr]
        
        for item in partitions:
            subtree = self.tdidt(item[1], attrRemaining, selectType)
            node[2].append(['value', item[0], subtree])
        
        return node

    def dt_get_subtree_classes(self, st, classDict={}):
        """Gets classes on all subtree paths and returns them in a list of dictionaries.
           [ {class0:count0}, {class1:count1}, ..., {classn:countn} ]."""
        nodeType = st[0]
        nodeVal  = st[1]
        if (nodeType == 'label'):
            if nodeVal not in classDict:
                classDict.update({nodeVal : 1})
            else:
                classDict[nodeVal] += 1
        else:
            for path in st[2]:
                classDict = self.dt_get_subtree_classes(path[2], classDict)
        return classDict
   
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

    def dt_classify(self, dt, instance, att='ROOT', selectType):
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
        # DAN
        if (selectType == 'categorical'):
            for child in nodeSubTree:
                childVal = child[1]
                if (instVal == childVal):
                    label = self.dt_classify(child[2], instance, nodeVal)
        else:
           leftChild = nodeSubTree[2]
           childValLeft = child[1][3:]
                

        # we couldn't find a path; majority vote on all possible subtrees
        if (label == 'NOCLASS'):
            for child in nodeSubTree:
                classes = self.dt_get_subtree_classes(child[2])
            values = list(classes.values())
            keys = list(classes.keys())
            label = keys[values.index(max(values))]
        return label

    def dt_build(self, table, attIndices, selectType):
        """Creates a decision tree for a data set and classifies instances
           according to the generated tree for each k in the k-fold cross validation
           Creates confusion matrices for the results and compares to HW3 classifiers."""       
        k = 10
        classLabels, actualLabels = [], []
        for curBin in range(k):
            train, test =  self.k_cross_fold_partition(table, k, self.classIndex, curBin)

            # build tree with training set
            self.decisionTree = self.tdidt(train, attIndices, selectType)

            # classify test set using tree
            for instance in test:
                classLabels.append(self.dt_classify(self.decisionTree, instance))
                actualLabels.append(instance[self.classIndex])
       
        return classLabels, actualLabels                
   
    def create_confusion_matrix(self, dataTitle, classLabels, actualLabels):
        """Creates confusion matrix for a given set of classified vs actual labels."""
        # create empty confusion matrix
        cfMatrix = [[0 for i in range(2 + len(self.uniqueClasses))] \
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
            cfMatrix[eIndex][aIndex] += 1

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
        
        for item in cfMatrix:
            print item
        # calculate recognition
        cfMatrix[0].append('Recognition Rate')
        for row in range(1, len(cfMatrix[0]) - 2):
            hits = float(cfMatrix[row][row])
            total = float(cfMatrix[row][-1])
            recognition = round(hits/total, 2)
            cfMatrix[row].append(str(recognition))
        cfMatrix[-1].append('NA')
        

        return cfMatrix
        
    def print_step_1(self):
        print '===================================================='
        print 'STEP 1: Decision Tree Classification (titanic.txt)'
        print '===================================================='
        table = self.table
        attIndices = [0, 1, 2]
        classLabels, actualLabels = self.dt_build(table, attIndices, 'categorical')
        cfMatrix = self.create_confusion_matrix('Titanic', classLabels, actualLabels)

        print tabulate(cfMatrix)
    
    def print_step_2(self):
        print '===================================================='
        print 'STEP 2: Decision Tree Classification (auto-data.txt)'
        print '        Categoical attributes (no split point)'
        print '===================================================='
        attIndices = [1, 4, 6]
        table = self.table

        # discretize mpg and weight, reset unique classes
        self.uniqueClasses = []
        for row in table:
            row[0] = self.discretize_mpg_doe(row[0])
            row[4] = self.discretize_weight_nhtsa(row[4])
            if row[self.classIndex] not in self.uniqueClasses:
                self.uniqueClasses.append(row[self.classIndex])
            self.uniqueClasses.sort()

        classLabels, actualLabels = self.dt_build(table, attIndices, 'categorical')
        cfMatrix = self.create_confusion_matrix('MPG', classLabels, actualLabels)

        print tabulate(cfMatrix)

    def print_step_3(self):
        print '===================================================='
        print 'STEP 3: Decision Tree Classification (auto-data.txt)'
        print '        Split point approach for continuous data'
        print '===================================================='
        attIndices = [1, 4, 6]
        table = self.table

        classLabels, actualLabels = self.dt_build(table, attIndices, 'continuous')
        # discretize class, actual, and unique labels
        self.uniqueClasses = []
        for i in range(len(classLabels)):
            classLabels[i] = self.discretize_mpg_doe(classLabels[i])
            actualLabels[i] = self.discretize_mpg_doe(actualLabels[i])
            if classLabels[i] not in self.uniqueClasses:
                self.uniqueClasses.append(classLabels[i])
            if actualLabels[i] not in self.uniqueClasses:
                self.uniqueClasses.append(actualLabels[i])
        self.uniqueClasses.sort()  
          
        self.dt_print(self.decisionTree)
        cfMatrix = self.create_confusion_matrix('MPG', classLabels, actualLabels)

        print tabulate(cfMatrix)
    	
def main():
    """Hello."""
    t = DecisionTreeClassifier('titanic.txt', -1)
    t.print_step_1()
    
    a = DecisionTreeClassifier('auto-data.txt', 0)
    a.print_step_2()
    
    a_sp = DecisionTreeClassifier('auto-data.txt', 0)
    a_sp.print_step_3()


if __name__ == "__main__":
    main()

