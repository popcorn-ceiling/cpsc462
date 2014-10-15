#!/usr/bin/env python
"""hw3.py:  Data mining assignment #3: Data classification and evaluation."""

__author__ = "Dan Collins and Miranda Myers"

import math
import random
import numpy 
import csv
import operator
import copy 
import warnings
import sys
import time

def processing():
    """This makes a super cool spinning thing for our cmd line while you wait."""
    print "    Calculating predictive accuracies...\\",
    syms = ['\\', '|', '/', '-']
    bs = '\b'

    for _ in range(5):
        for sym in syms:
            sys.stdout.write("\b%s" % sym)
            sys.stdout.flush()
            time.sleep(.25)
    print '\bDone!'

class DataClassification:
    """FIXME."""

    def __init__(self, ):
        """Constructor creates a table of pre-cleaned data read from a file."""
        self.__table = self.read_csv('auto-data.txt')

    def get_table_len(self):
        """Accessor for class data table length. Used in main for seed."""
        return len(self.__table)

    def read_csv(self, filename):
        """Reads in a csv file and returns a table as a list of lists (rows)."""
        the_file = open(filename)
        the_reader = csv.reader(the_file, dialect='excel')
        table = []
        for row in the_reader:
            if len(row) > 0:
                table.append(row)
        return table

    def get_column_as_floats(self, table, index):
        """Returns all non-null values in table for given index."""
        vals = []
        for row in table:
            if row[index] != "NA":
                vals.append(float(row[index]))
        return vals    
    
    def average(self, vals):
        """Finds the average of a column (array) of values."""
        if len(vals) != 0:
            return round(float(sum(vals)/len(vals)), 2)
        else:
            return 0

    def calculate_least_squares_lr(self, xs, ys):
        """Calculates the slope (m) and y-intercept (b) of the linear \
           regression line using the least squares method."""
        xAvg = self.average(xs)
        yAvg = self.average(ys)

        #Calculate m, slope of line
        mTop = 0
        mBot = 0
        for i in range(len(xs)):
            mTop += ((xs[i] - xAvg)*(ys[i] - yAvg)) 
            mBot += (xs[i] - xAvg)**2
        m = float(mTop / mBot)

        #Calculate b, y intercept of line
        b = yAvg - (m * xAvg)

        return m, b
    
    def calculate_covariance(self, xs, ys):
        """Calcualtes the covariance given a set of (x,y) values."""
        xAvg = self.average(xs)
        yAvg = self.average(ys)
           
        cov_sum = 0
        for i in range(len(xs)):
            cov_sum += (xs[i] - xAvg)*(ys[i] - yAvg)

        return float(cov_sum / len(xs))

    def calculate_corr_coefficient(self, xs, ys, cov):
        """Calculates the correlation coefficient given a set of (x,y) \
           values and the covariance of the data set."""
        stdx = numpy.std(xs)
        stdy = numpy.std(ys)
    
        return float(cov/(stdx*stdy))
        
    def classify_mpg_lr(self, trainingSet, index, x):
        """FIXME."""
        #Get the list of values to be compared with mpg (weight for this program)
        xs = self.get_column_as_floats(trainingSet, index)
        
        #Get the list of mpg values
        ys = self.get_column_as_floats(trainingSet, 0)
        
        #Calculate linear regression values for mpg and given variable (weight)
        m, b = self.calculate_least_squares_lr(xs, ys)
        
        #Calculate the predicted value for the given x (a weight value)
        y = m * float(x) + b
        
        #Classify based on department of energy ratings
        classification = self.classify_mpg_DoE(y)
        
        return classification
        
    def classify_mpg_DoE(self, y): 
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
        #FIXME
        #Implement error checking else statement
        #Ask bowers about edges of bins
                
        return rating

    def test_random_instances_step1(self, seed):
        """Test step 1 classifier on 5 random instances."""
        print '==========================================='
        print 'STEP 1: Linear Regression MPG Classifier'
        print '==========================================='
        
        table = copy.deepcopy(self.__table)
        weights = self.get_column_as_floats(table, 4)
                    
        testValues = []
        printIndices = []
        for rand_i in seed:
            val = weights[rand_i]
            
            printIndices.append(rand_i)
            testValues.append(val)
                        
        for i in range(len(testValues)):
            classification = self.classify_mpg_lr(table, 4, testValues[i])
            instance = table[printIndices[i]]
            actual = self.classify_mpg_DoE(instance[0])
            print '    instance:', ", ".join(instance)
            print '    class:', str(classification) + ',', 'actual:', actual
        print
            
    def normalize(self, xs):
        """FIXME."""
        minval = min(xs)
        maxval = max(xs)
        maxmin = (maxval - minval) * 1.0
        return [(x - minval) / maxmin for x in xs] 
            
    def normalize_table(self, table, indices):
        """Normalizes table (list of lists) based for attribute indices (list)."""
        table = copy.deepcopy(self.__table)

        # normalize attributes that we care about
        normalizedColumns = []
        for i in indices:
            normalizedColumns.append(self.get_column_as_floats(table, i))
            normalizedColumns[-1] = self.normalize(normalizedColumns[-1])

        # normalize data set
        normalizedTable = []
        for i in range(len(table)):
            newRow = table[i][:]
            for j in range(len(normalizedColumns)):
                newRow[indices[j]] = str(normalizedColumns[j][i])
            normalizedTable.append(newRow)
        
        return normalizedTable

    def calculate_euclidean_distance(self, row, instance, indices):
        """FIXME."""
        distance_sum = 0.0
        for i in indices:
            distance_sum += (float(row[i]) - float(instance[i])) ** 2
        
        return math.sqrt(distance_sum)
    
    def k_nn_classifier(self, trainingSet, indices, instance, k, classIndex):
        """Classifies an instance using k nearest neightbor method. 
           Assumes data provided is already normalized."""
        # Create list of rows with corresponding distances
        row_distances = []
        for row in trainingSet:
            row_distances.append([self.calculate_euclidean_distance( \
                                  row, instance, indices), row])
        
        #Sort the list to select the closest k distances
        row_distances.sort()
        return self.select_class_label(row_distances[0:k], classIndex)
    
    def select_class_label(self, closest_k, class_index):
        '''Select the class label for the nearest k neighbors. '''
        # Assign points to the nearest k neighbors
            # Points start at 1 for the farthest away and 
            # increment by one up to the nearest neighbor
        labels = []
        points = []
        for i in range(len(closest_k) - 1, -1, -1):
            labels.append(closest_k[i][class_index+1][0])
            points.append(i+1)
        
        # Create a dictionary of the labels with corresponding total points 
        pointLabelDict = {}
        for i in range(len(labels)):
            if labels[i] not in pointLabelDict.keys():
                pointLabelDict.update({labels[i] : points[i]})
            else:
                pointLabelDict[labels[i]] += points[i]
                
        #Find key(s) with the max total points
        maxPoints = max(pointLabelDict.values())
        maxKeys = [x for x,y in pointLabelDict.items() if y == maxPoints]
        
        # TODO implement tie breaker
        return self.classify_mpg_DoE(maxKeys[0])
   
    def test_random_instances_step2(self, seed):
        """Test step 2 classifier on 5 random instances."""
        print '==========================================='
        print 'STEP 2: k=5 Nearest Neighbor MPG Classifier'
        print '==========================================='

        k = 5
        classIndex = 0 # mpg
        indices = [1, 4, 5] # cylinders, weight, acceleration
        table = self.normalize_table(self.__table, indices)    
        trainingSet = table # training and test are same for this step
        testSet = table

        # classify k=5 random instances
        for rand_i in seed:
            instance = testSet[rand_i]
            origInstance = self.__table[rand_i]
            classification = self.k_nn_classifier(trainingSet, \
                             indices, instance, k, classIndex)

            actual = self.classify_mpg_DoE(instance[0])
            print '    instance:', ", ".join(origInstance)
            print '    class:', str(classification) + ',', 'actual:', actual
        print
    
    def discretize_weight_nhtsa(self, strWeight):
        """Discretize a given` weight according to NHTSA vehicle size ranking."""
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

    def gaussian(self, x, mean, sdev):
        """FIXME."""
        first, second = 0, 0
        if sdev > 0:
            first = 1 / (math.sqrt(2 * math.pi) * sdev)
            second = math.e ** (-((x - mean) ** 2) / (2 * (sdev ** 2)))
        return first * second
        
    def categorize_weight(self, table):
        """Return the dataset table with all weight values categorized."""
        for row in table:
            row[4] = self.discretize_weight_nhtsa(row[4])
        return table    
        
    def calculate_probabilities(self, columnIndex, table):
        """Returns the probability of each value occurring in a column."""
        column = self.get_column_as_floats(table, columnIndex)
        sortedColumn = sorted(column)
        totalValues = len(sortedColumn)
        
        values, probabilities = [], []
        for value in sortedColumn:
            if value not in values:
                values.append(value)
                probabilities.append(1)
            else:
                probabilities[-1] += 1
        
        for i in range(len(probabilities)):
            probabilities[i] /= float(totalValues)
        
        return values, probabilities 
        
    def calculate_pX(self, indices, instance, table):
        """Calculate the probability of an instance X given a dataset table.
           Requires values and instance[i] to be of same type!"""
        # For each index, calculate its probability for the given instance
        # assumes strings for comparison        
        pX = 1
        # FIXME TODO fix types here
        for i in indices:
            values, probabilities = self.calculate_probabilities(i, table)
            if instance[i] not in values:
                probability = 0.0
            else:
                probability = probabilities[values.index(instance[i])]
            # Multiply all probabilities together
            pX *= probability
        return pX
    
    def calculate_pXCi(self, classIndex, instance, table, classNames, attrIndices):
        """FIXME."""
        pXCi = []
        for i in range(len(classNames)):
            newList = self.partition_classes(classIndex, classNames[i], table)
            pXC = self.calculate_pX(attrIndices, instance, newList)
            pXCi.append(float(pXC))

        return pXCi
    
    def calculate_pXCi_ctns(self, classIndex, instance, table, classNames, attrIndices):
        """FIXME."""
        pXCi = []
        for i in range(len(classNames)):
            newList = self.partition_classes(classIndex, classNames[i], table)
            pXC = 1 # replaces pXCi for continuous attributes
            for attr in attrIndices:
                x = float(instance[attr])
                column = self.get_column_as_floats(newList, attr)
                uc = self.average(column)
                std = numpy.std(column)
                pXC *= self.gaussian(x, uc, std)
            pXCi.append(float(pXC))

        return pXCi

    def partition_classes(self, classIndex, className, table):
        """Given a class name and index, return a table of instances that contain that class."""
        classPartition = []
        for row in table:
            if float(row[classIndex]) == float(className):
                classPartition.append(row)
        return classPartition

    def naive_bayes_i(self, instance, classIndex, attrIndices, trainingSet):
        """Classifies an instance using the naive bayes method."""
        fInstance = instance[:]
        for i in attrIndices:
            fInstance[i] = float(instance[i])
        pCiLabel, pCi = self.calculate_probabilities(classIndex, trainingSet)
        pXCi = self.calculate_pXCi(classIndex, fInstance, trainingSet, pCiLabel, attrIndices)
        pCX = []
        for i in range(len(pCi)):
            pCX.append((pXCi[i]*pCi[i]))
        
        return pCiLabel[pCX.index(max(pCX))]
    
    def naive_bayes_ii(self, instance, classIndex, attrIndices, trainingSet):
        """Classifies an instance using the naive bayes method. Differs from 
           naive_bayes_i in that it assumes a guassian distribution for the
           probability of an instance in a class."""
        fInstance = instance[:]
        for i in attrIndices:
            fInstance[i] = float(instance[i])
        pCiLabel, pCi = self.calculate_probabilities(classIndex, trainingSet)
        pXCi = self.calculate_pXCi_ctns(classIndex, fInstance, trainingSet, pCiLabel, attrIndices)
        pCX = []
        for i in range(len(pCi)):
            pCX.append((pXCi[i]*pCi[i]))
        
        return pCiLabel[pCX.index(max(pCX))]
        
    def test_random_instances_step3(self, seed):
        """FIXME."""
        print '==========================================='
        print 'STEP 3: Naive Bayes MPG Classifiers'
        print '==========================================='

        attrIndices = [1, 4, 6] # cylinders, weight, year
        classIndex = 0 # mpg
        table = copy.deepcopy(self.__table)
        table = self.categorize_weight(table)
        for row in table:
            row[0] = str(self.classify_mpg_DoE(row[0]))

        print 'Naive Bayes I:'
        for rand_i in seed:
            instance = table[rand_i]
            origInstance = self.__table[rand_i]
            actual = instance[0]
            classification = self.naive_bayes_i(instance, classIndex, attrIndices, table)
            print '    instance:', ", ".join(origInstance)
            print '    class:', str(int(classification)) + ',', 'actual:', actual

        print 'Naive Bayes II:'
        for rand_i in seed:
            instance = table[rand_i]
            origInstance = self.__table[rand_i]
            actual = instance[0]
            classification = self.naive_bayes_ii(instance, classIndex, attrIndices, table)
            print '    instance:', ", ".join(origInstance)
            print '    class:', str(int(classification)) + ',', 'actual:', actual
        print

    def holdout_partition(self, table):
        """FIXME."""
        # randomize the table
        randomized = table # table passed is already a copy
        n = len(table)
       
        for i in range(n):
            # pick an index to swap
            j = random.randint(0, n-1) # random int [0,n-1] inclusive
            randomized[i], randomized[j] = randomized[j], randomized[i]
        # return train and test sets
        n0 = (n * 2)/3
        return randomized[0:n0], randomized[n0:]
            
    def calculate_predacc(self, classifiedLabels, actualLabels, numInst):
        """Calculates predictive accuracy given two list of guessed 
           and actual labels, and the number of instances."""
        correctClassificationCount = 0
        for i in range(len(classifiedLabels)):
            actual = str(actualLabels[i])
            expected = str(classifiedLabels[i])
            if actual == expected:
                correctClassificationCount += 1
                
        return correctClassificationCount / float(numInst)

    def calculate_std_error(self, predacc_estimate, n):
        """Calculates the standard error given a predictive accuracy estimate 
           and a test set size."""
        stdError = math.sqrt( \
                (predacc_estimate * (1 - predacc_estimate)) / float(n))
        return round(stdError, 2)
        
        # TODO what is this?
        # true predictive accuracy lies in interval: 
        # p +- Zcl * stdError <- use table in book
        
    def accuracy_random_subsampling(self, repeatNum, whichClassifier):
        """Calculate accuracy using random subsampling by repeating the 
           holdout method k times. Also returns test set size as 2nd return."""
        k = 5 # k in context of k-nn, not k subsamples
        classIndex = 0
        indices = [1, 4, 5]
        table = copy.deepcopy(self.__table)
        table = self.categorize_weight(table)
        for row in table:
            row[0] = str(self.classify_mpg_DoE(row[0]))
        normTable = self.normalize_table(table, indices)    

        predAccs = []
        for i in range(repeatNum):
            classLabels, actualLabels = [], []
            # partition dataset
            if whichClassifier == 1 or whichClassifier == 2:
                tableUsed = table
            else:
                tableUsed = normTable
            trainingSet, testSet = self.holdout_partition(tableUsed)
            # select classifier
            for instance in testSet:
                if whichClassifier == 0:
                    classLabels.append(float(self.classify_mpg_lr(trainingSet, 4, instance[4])))
                    actualLabels.append(float(self.classify_mpg_DoE(instance[0])))
                elif whichClassifier == 1:
                    label = self.naive_bayes_i(instance, classIndex, indices, trainingSet)
                    classLabels.append(float(self.classify_mpg_DoE(label)))
                    actualLabels.append(float(instance[0]))
                elif whichClassifier == 2:
                    label = self.naive_bayes_ii(instance, classIndex, indices, trainingSet)
                    classLabels.append(float(self.classify_mpg_DoE(label)))
                    actualLabels.append(float(instance[0]))
                elif whichClassifier == 3:
                    classLabels.append(float(self.k_nn_classifier(trainingSet, indices, \
                                             instance, k, classIndex)))
                    actualLabels.append(float(self.classify_mpg_DoE(instance[0])))
                else:
                    print 'error: unknown classifier specified'
                    exit(-1)
            # calculate predictive accuracy 
            predAccs.append(self.calculate_predacc(classLabels, actualLabels, len(testSet)))
       
        # accuracy estimate is the average of the accuracy of each iteration
        avgPredAcc = round(sum(predAccs) / len(predAccs), 2)
        stderr = self.calculate_std_error(avgPredAcc, len(testSet))
        return avgPredAcc, stderr
    
    def evaluate_classifiers_step4(self):
        """Evaluates predictive accuracy of classifiers used so far using 
           predictive accuracy and standard error."""
        print '==========================================='
        print 'STEP 4: Predictive Accuracy'
        print '==========================================='
        k = 10
        
        print '    Random Subsample (k=10, 2:1 Train/Test)'
        #processing()
        predacc_lr, stderr_lr     = self.accuracy_random_subsampling(k, 0)
        print '        Linear Regression      : p =', predacc_lr, '+-', stderr_lr 
        predacc_nbi, stderr_nbi   = self.accuracy_random_subsampling(k, 1)
        print '        Naive Bayes I          : p =', predacc_nbi, '+-', stderr_nbi 
        predacc_nbii, stderr_nbii = self.accuracy_random_subsampling(k, 2)
        print '        Naive Bayes II         : p =', predacc_nbii, '+-', stderr_nbii 
        predacc_knn, stderr_knn   = self.accuracy_random_subsampling(k, 3)
        print '        Top-5 Nearest Neighbor : p =', predacc_knn, '+-', stderr_knn 
        
    def k_cross_fold_partition(self, table, k, classIndex, curBin):
        """FIXME."""
        # get classes
        classNames = []
        for row in table:
            if row[classIndex] not in classNames:
                classNames.append(row[classIndex])
        classNames.sort()
        # partition dataset - each subset contains rows with a unique class
        dataPartition = []
        for i in range(len(classNames)):
            dataPartition.append(self.partition_classes(classIndex, classNames[i], table))

        # distribute paritions roughly equally
        kPartitions = [[] for _ in range(k)]
        for partition in dataPartition:
            for i in range(len(partition)):
                kPartitions[i%k].append(partition[i])
        
        # return training and test set
        testSet = kPartitions[curBin]
        for i in range(k):
            if i != curBin:
                trainingSet.append(kPartitions)
        return trainingSet, testSet
 
def main():
    # mpg, cylinders, displacement, horsepower, weight
    # acceleration, model year, origin, car name       
    warnings.simplefilter("error")
    d = DataClassification()
 
    # generate seed - 5 random row indices from table
    seed = []
    for i in range(5):
        rand = random.randint(0, d.get_table_len() - 1)
        while i in seed:
            rand = random.randint(0, d.get_table_len() - 1)
        seed.append(rand)

    d.test_random_instances_step1(seed)
    d.test_random_instances_step2(seed)
    d.test_random_instances_step3(seed)
    d.evaluate_classifiers_step4()


if __name__ == "__main__":
    main()
    


