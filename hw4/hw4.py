import copy
import csv
import random
from math import log
from tabulate import tabulate

class DecisionTreeClassifier:

    def __init__(self, fileName, classIndex):
        self.table = self.read_csv(fileName)
        self.attrNames = self.table.pop(0)
        self.classIndex = classIndex
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
        """Partition a dataset into training and test by splitting into K folds (bins)."""
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
        
        # partition dataset - each subset contains rows with a unique class
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
        
    """
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
            Enew += (Dj / D) * EDj
        return Enew
    """
        
    def calculate_Enew(self, partitions, instances):
        '''Calculate Enew given a list of partitions for a single attribute.'''
        Enew = 0
        for partition in partitions:
            EDj = self.calculate_entropy(partition[1])
            Dj = len(partition[1])
            D = len(instances)
            Enew += (Dj / D) * EDj
        return Enew
            
    def find_smallest_Enew(self, instances, attIndices, partitionType):
        '''Find which attribute partition gives the smallest attribute value - the
            smallest amount of information needed to classify an instance.'''
        EnewList = []
        for attIndex in attIndices:
            # Calculate Enew for the instances partitioned on the given attribute
            if partitionType == 'categorical':
                partitions = self.partition_instances(instances, attIndex)
            elif partitionType == 'continuous':
                partitions = self.partition_on_split_pt(instances, attIndex)
            else:
                print 'Pick a correct partition type next time!!!'
                exit()
            Enew = self.calculate_Enew(partitions, instances)
            EnewList.append(Enew)
        
        smallestEnew = min(EnewList)
        attIndex = attIndices[EnewList.index(smallestEnew)]
        
        # Return the attribute index to use for partitioning
        return attIndex
    
    def select_split_point(self, instances, attIndices):
        '''Select the split point that minimizes Enew.'''
        EnewList = []
        
        # Sort the instances on the attribute index
        
        
        
        # Return the attribute index to split on
        
        
    def partition_on_split_pt(self, instances, splitPoint):
        '''FIXME.'''
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
        attIndex = self.find_smallest_Enew(instances, attIndices, selectionType)
        return attIndex
        
        #elif selectionType == 'categorical':
            #pass #TODO implement
            
        #else:
        #    print 'Chosen attribute selection method is not valid'
        #    exit()
    
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
        # No more attributes to partition
        if len(attIndices) == 0:
            stats = self.partition_stats(instances)
            label = self.resolve_clash(stats)
            return ['label', label]
        # Only class labels that are the same
        elif self.in_same_class(instances, self.classIndex):
            label = instances[0][self.classIndex]
            return ['label', label]

        # No more instances to partition
        if len(instances) == 0:
            return

        # At each step select an attribute and partition data 
        attr = self.select_attribute(instances, attIndices, selectType)
        partitions = self.partition_instances(instances, attr)

        node = ['attribute', self.attrNames[attr], []]
        attrRemaining = [item for item in list(attIndices) if item != attr]
        
        for item in partitions:
            subtree = self.tdidt(item[1], attrRemaining, selectType)
            node[2].append(['value', item[0], subtree])
        
        return node

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
        label = 'ERROR: NOT CLASSIFIED'
        # search the child nodes for matches with our instance
        for child in nodeSubTree:
            childVal = child[1]
            if (instVal == childVal):
                label = self.dt_classify(child[2], instance, nodeVal)

        return label

    def print_dt(self, dt, level=1):
        """Debug print function for trees."""
        if len(dt) > 0:
            if len(dt) > 2:
                print '|' + (level * '---'), dt[0],':',dt[1]
                for item in dt[2]:
                    print '|' +  ((level + 1) * '---'), item[0], ':', item[1]
                    self.print_dt(item[2], level + 2)
            else:
                print '|' + (level * '---'), dt[0],':',dt[1]
                return

    def dt_build(self, attIndices, selectType):
        """Creates a decision tree for a data set and classifies instances
           according to the generated tree for each k in the k-fold cross validation
           Creates confusion matrices for the results and compares to HW3 classifiers."""
        k = 10
        table = self.table
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
   
    def confusion_matrix_titanic(self, dataSet, classLabels, actualLabels):
        """Creates confusion matrix for binary classification of titanic: suvived."""
        # Calculate true positives, false negatives, false positives, true negatives
        TP, FN, FP, TN = 0, 0, 0, 0
        
        for i in range(len(classLabels)):
            if classLabels[i] == 'yes' and actualLabels[i] == 'yes':
                TP +=1
            elif classLabels[i] == 'no' and actualLabels[i] == 'yes':
                FN += 1
            elif classLabels[i] == 'yes' and actualLabels[i] == 'no':
                FP += 1
            else:
                TN += 1
        
        P = TP + FN
        N = FP + TN
        PplusN = P + N
        Ppred = TP + FP
        Npred = FN + TN
         
        confusionMatrix = []
        confusionMatrix.append([dataSet, 'yes', 'no', 'Total'])
        confusionMatrix.append(['yes', str(TP), str(FN), str(P)])  
        confusionMatrix.append(['no', str(FP), str(TN), str(N)])
        confusionMatrix.append(['Total', str(Ppred), str(Npred), str(PplusN)])
        
        return confusionMatrix
        
    def print_step_1(self):
        attIndices = [0, 1, 2]
        classLabels, actualLabels = self.dt_build(attIndices, 'categorical')
        
        confusionMatrix = self.confusion_matrix_titanic('Titanic', classLabels, actualLabels)

        print '==================================================='
        print 'STEP 1: Decision Tree Classification (titanic.txt)'
        print '==================================================='
        print tabulate(confusionMatrix)
    


    	
def main():
    """Hello."""
    t = DecisionTreeClassifier('titanic.txt', -1)
    t.print_step_1()
    
    t2 = DecisionTreeClassifier('auto-data.txt', -1)
    

if __name__ == "__main__":
    main()

