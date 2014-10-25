import copy
import csv
import random
from math import log
import tabulate

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
        labels, probabilities = self.calculate_pi(self.classIndex, instances)
        
        # Iterate through the class labels to calculate entropy
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
        
        # Calculate Enew
        Enew = 0
        for partition in partitions:
            EDj = self.calculate_entropy(partition[1])
            Dj = len(partition[1])
            D = len(instances)
            Enew += (Dj / D) * EDj
        return Enew
            
    def find_smallest_Enew(self, instances, attIndices, classIndex):
        '''.'''
        EnewList = []
        for attIndex in attIndices:
            Enew = self.calculate_Enew(instances, attIndex)
            EnewList.append(Enew)
        
        smallestEnew = min(EnewList)
        attIndex = attIndices[EnewList.index(smallestEnew)]
        return attIndex
                
    def select_attribute(self, instances, attIndices, selectionType):
        '''Returns attribute index to partition on using chosen selection method.'''
        
        if selectionType == 'entropy':
            attIndex = self.find_smallest_Enew(instances, attIndices, self.classIndex)
            return attIndex
        elif selectionType == 'split point':
            pass #TODO implement
            
        else:
            print 'Chosen attribute selection method is not valid'
            exit()
    
    def resolve_clash(self, statDict):
        '''.'''
        # TODO what happens in 50/50?
        values = list(statDict.values())
        keys = list(statDict.keys())
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

    def dt_classify(self, dt, instance, attIndices):
        """Classifies an instance using a decision tree passed to it."""
        
        return 'yes'

    def print_dt(self, dt, treestr, level=1):
        """Debug print function for trees."""
        if len(dt) > 0:
            if len(dt) > 2:
                print treestr + (level * '---'), dt[0],':',dt[1]
                for item in dt[2]:
                    print treestr +  ((level + 1) * '---'), item[0], ':', item[1]
                    self.print_dt(item[2], treestr, level + 2)
            else:
                print treestr + (level * '---'), dt[0],':',dt[1]
                return
                

    def decisiontree(self, attIndices, selectType):
        """Creates a decision tree for titanic.txt and classifies instances
           according to the generated tree for each k in the k-fold cross validation
           Creates confusion matrices for the results and compares to HW3 classifiers."""
        # TODO Do we have to do the comparison to HW3 classifiers in the code, or just the log
        
        k = 10
        table = self.table
        for curBin in range(k):
            train, test =  self.k_cross_fold_partition(table, k, self.classIndex, curBin)

            # build tree with training set
            self.decisiontree = self.tdidt(train, attIndices, selectType)
            treestr = '|'
            self.print_dt(self.decisiontree, treestr)    

            # classify test set using tree
            classLabels, actualLabels = [], []
            for instance in test:
                classLabels.append(self.dt_classify(self.decisiontree, instance, attIndices))
                actualLabels.append(instance[self.classIndex])
            
            return classLabels, actualLabels                
   
    def confusion_matrix_titanic(self, classLabels, actualLabels):
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
        confusionMatrix.append(['', 'yes', 'no', 'Total'])
        confusionMatrix.append(['yes', str(TP), str(FN), str(P)])  
        confusionMatrix.append(['no', str(FP), str(TN), str(N)])
        confusionMatrix.append(['Total', str(Ppred), str(Npred), str(PplusN)])
        
        return confusionMatrix
        
    def print_step_1(self):
        attIndices = [0, 1, 2]
        classLabels, actualLabels = self.decisiontree(attIndices, 'entropy')
        #print 'c', classLabels
        #print
        #print 'a', actualLabels
        #confusionMatrix = self.confusion_matrix_titanic(classLabels, actualLabels)
        #print tabulate(confusionMatrix)
    

def main():
    """Hello."""
    t = DecisionTreeClassifier('titanic.txt', -1)
    t.print_step_1()

if __name__ == "__main__":
    main()

