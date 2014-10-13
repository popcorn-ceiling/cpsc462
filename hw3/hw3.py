#!/usr/bin/env python
"""hw3.py:  Data mining assignment #3: Data classification and evaluation."""

__author__ = "Dan Collins and Miranda Myers"

import math
import random
import numpy 
import csv
import operator
import copy 

class DataClassification:
    """FIXME."""

    def __init__(self, ):
        """Constructor creates a table of pre-cleaned data read from a file."""
        self.__table = self.read_csv('auto-data.txt')

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
        
    def classify_mpg_lr(self, table, index, x):
        """FIXME."""
        #Get the list of values to be compared with mpg (weight for this program)
        xs = self.get_column_as_floats(table, index)
        
        #Get the list of mpg values
        ys = self.get_column_as_floats(table, 0)
        
        #Calculate linear regression values for mpg and given variable (weight)
        m, b = self.calculate_least_squares_lr(xs, ys)
        
        #Calculate the predicted value for the given x (a weight value)
        y = m * x + b
        
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

    def test_random_instances_step1(self):
        """Test step 1 classifier on 5 random instances."""
        print '==========================================='
        print 'STEP 1: Linear Regression MPG Classifier'
        print '==========================================='
        
        table = copy.deepcopy(self.__table)
        weights = self.get_column_as_floats(table, 4)
                    
        testValues = []
        printIndices = []
        for i in range(5):
            rand_i = random.randint(0, len(weights)-1)
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
            
    def calculate_euclidean_distance(self, row, instance, indices):
        """FIXME."""
        distance_sum = 0.0
        for i in indices:
            distance_sum += (float(row[i]) - float(instance[i])) ** 2
        
        return math.sqrt(distance_sum)
    
    def k_nn_classifier(self, table, trainingSet, testSet, indices, instance, k, class_index):
        """FIXME."""
        row_distances = []
        
        columns = []
        for i in indices:
            column = self.get_column_as_floats(table, i)
            columns.append(column)
   
        # Normalize the training set
        normalizedTrainingSet = []
        for j in range(len(trainingSet)):
            newRow = trainingSet[j][:]
            for i in range(len(columns)):
                normalizedColumn = self.normalize(columns[i])
                newRow[indices[i]] = normalizedColumn[j]
            normalizedTrainingSet.append(newRow)
            
            
        # Normalize the test set to normalize the instance
        normalizedTestSet = []
        for j in range(len(testSet)):
            newRow = testSet[j][:]
            for i in range(len(columns)):
                normalizedColumn = self.normalize(columns[i])
                newRow[indices[i]] = normalizedColumn[j]
            normalizedTestSet.append(newRow)
         
        instanceIndex = testSet.index(instance)  
        normalizedInstance = instance[:]
        
        for i in range(len(indices)):
            normalizedColumn = self.normalize(columns[i])
            normalizedInstance[indices[i]] = normalizedColumn[instanceIndex]
            

        # Create list of rows with corresponding distances
        for row in normalizedTrainingSet:
            row_distances.append([self.calculate_euclidean_distance(row, normalizedInstance, indices), row])
        
        #Sort the list to select the closest k distances
        row_distances.sort()
        return self.select_class_label(row_distances[0:k], class_index)
    
    def select_class_label(self, closest_k, class_index):
        '''Select the class label for the nearest k neighbors. '''
        # Assign points to the nearest k neighbors
            # Points start at 1 for the farthest away and increment by one up to the 
            # nearest neighbor
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
        
        # implement tie breaker
    
        return self.classify_mpg_DoE(maxKeys[0])
   
    def test_random_instances_step2(self):
        """Test step 2 classifier on 5 random instances."""
        print '==========================================='
        print 'STEP 2: k=5 Nearest Neighbor MPG Classifier'
        print '==========================================='

        k = 5
        indices = [1, 4, 5] # cylinders, weight, acceleration
        class_index = 0 # mpg
        table = copy.deepcopy(self.__table)
        trainingSet = copy.deepcopy(self.__table)
        testSet = copy.deepcopy(self.__table)

        for i in range(k):
            rand_i = random.randint(0, len(table)-1)
            instance = table[rand_i]
            classification = self.k_nn_classifier(table, trainingSet, testSet, indices, instance, k, class_index)

            actual = self.classify_mpg_DoE(instance[0])
            print '    instance:', ", ".join(instance)
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
        categorizedTable = copy.deepcopy(table)
        for row in table:
            row[4] = self.discretize_weight_nhtsa(row[4])
        return categorizedTable     
        
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
        """FIXME."""
        # For each index, calculate its probability for the given instance
        # assumes strings for comparison        
        pX = 1
        for i in indices:
            values, probabilities = self.calculate_probabilities(i, table)
            # FIXME TODO fix types here
            if str(float(instance[i])) not in str(values):
                probability = 0.0
            else:
                probability = probabilities[values.index(float(instance[i]))]
            # Multiply all probabilities together
            pX *= probability
        
        return pX
    
    def calculate_pXCi(self, classIndex, instance, table, classNames, attrIndices):
        """FIXME."""
        pXCi = []
        for i in range(len(classNames)):
            newList = self.partition_classes(classIndex, str(int(classNames[i])), table)
            pXC = self.calculate_pX(attrIndices, instance, newList)
            pXCi.append(float(pXC))

        return pXCi
    
    def calculate_pXCi_ctns(self, classIndex, instance, table, classNames, attrIndices):
        """FIXME."""
        pXCi = []
        for i in range(len(classNames)):
            newList = self.partition_classes(classIndex, str(int(classNames[i])), table)
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
            if row[classIndex] == className:
                classPartition.append(row)
        return classPartition

    def naive_bayes_i(self, instance, classIndex, attrIndices, table):
        """FIXME."""
        pX = self.calculate_pX(attrIndices, instance, table)
        pCiLabel, pCi = self.calculate_probabilities(classIndex, table)
        pXCi = self.calculate_pXCi(classIndex, instance, table, pCiLabel, attrIndices)
        pCX = []
        for i in range(len(pCi)):
            pCX.append((pXCi[i]*pCi[i]))
        return pCiLabel[pCX.index(max(pCX))]
    
    def naive_bayes_ii(self, instance, classIndex, attrIndices, table):
        """FIXME."""
        pX = self.calculate_pX(attrIndices, instance, table)
        pCiLabel, pCi = self.calculate_probabilities(classIndex, table)
        pXCi = self.calculate_pXCi_ctns(classIndex, instance, table, pCiLabel, attrIndices)
        pCX = []
        for i in range(len(pCi)):
            pCX.append((pXCi[i]*pCi[i]))
        return pCiLabel[pCX.index(max(pCX))]
        
    def test_random_instances_step3(self):
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

        rand_i = []
        for i in range(5):
            rand = random.randint(0, len(table)-1)
            while i in rand_i:
                rand = random.randint(0, len(table)-1)
            rand_i.append(rand)

        print 'Naive Bayes I:'
        for i in rand_i:
            instance = table[i]
            actual = instance[0]
            classification = self.naive_bayes_i(instance, classIndex, attrIndices, table)
            print '    instance:', ", ".join(instance)
            print '    class:', str(int(classification)) + ',', 'actual:', actual

        print 'Naive Bayes II:'
        for i in rand_i:
            instance = table[i]
            actual = instance[0]
            classification = self.naive_bayes_ii(instance, classIndex, attrIndices, table)
            print '    instance:', ", ".join(instance)
            print '    class:', str(int(classification)) + ',', 'actual:', actual
        print

    def holdout_partition(self, table):
        """FIXME."""
        # randomize the table
        randomized = copy.deepcopy(table) # copy the table
        n = len(table)
       
        for i in range(n):
            # pick an index to swap
            j = random.randint(0, n-1) # random int [0,n-1] inclusive
            randomized[i], randomized[j] = randomized[j], randomized[i]
        # return train and test sets
        n0 = (n * 2)/3
        return randomized[0:n0], randomized[n0:]
            
    def calculate_predictive_accuracy_knn(self, table, trainingSet, indices, k, class_index, testSet):
        """FIXME."""
        correctClassificationCount = 0
        numTestInstances = len(testSet)
        
        # use classifier to predict the classification for the instances in the test set
        for instance in testSet:
            classLabel = self.k_nn_classifier(table, trainingSet, testSet, indices, instance, k, class_index)
            actualLabel = self.classify_mpg_DoE(instance[0])
            if str(classLabel) == str(actualLabel):
                correctClassificationCount += 1
                
        #Calculate predictive accuracy
        predictiveAccuracy = correctClassificationCount / float(numTestInstances)
        print 'PREDACC', predictiveAccuracy
        return predictiveAccuracy
        

    def calculate_std_error(self, predict_acc_estimate, n):
        """FIXME."""
        stdError = math.sqrt(predict_acc_estimate * (1 - predict_acc_estimate) / float(n))
        return stdError
        #true predictive accuracy lies in interval: p +- Zcl * stdError <- use table in book
        return randomized[0:n0], randomized[n0:]
        
    def random_subsampling_accuracy_knn(self, repeatNum):
        """Calculate accuracy using random subsampling by repeating the holdout method \
            k times."""
        #accuracy estimate is the average of the accuracy of each iteration
        #classifier is used to predict the classification for the instances in the test set
        table = copy.deepcopy(self.__table)
        indices = [1, 4, 5]
        k = 5
        classIndex = 0
        predictiveAccuracies = []
        for i in range(repeatNum):
            trainingSet, testSet = self.holdout_partition(table)
            predictiveAccuracy = self.calculate_predictive_accuracy_knn(table, trainingSet, indices, k, classIndex, testSet)
            predictiveAccuracies.append(predictiveAccuracy)
        
        # Calculate the average predictive accuracy    
        avgPredictiveAccuracy = sum(predictiveAccuracies) / len(predictiveAccuracies)
        
        print avgPredictiveAccuracy
        return avgPredictiveAccuracy

        # return train and test sets
        n0 = (n * 2)/3
        return randomized[0:n0], randomized[n0:]
    
    def test_dan(self):
        table = copy.deepcopy(self.__table)
        for row in table:
            row[0] = str(self.classify_mpg_DoE(row[0]))
        parts = self.k_cross_fold_partition(table, 10, 0)
       # i = 1
       # for item in parts:
       #     print 'HEYYYYAAA', len(item), i
       #     for beef in item:
       #         print beef
       #     i += 1

    def k_cross_fold_partition(self, table, k, classIndex):
        """FIXME."""
        # get classes
        classNames = []
        for row in table:
            if row[classIndex] not in classNames:
                classNames.append(row[classIndex])
        classNames.sort()
        print classNames
        # partition dataset - each subset contains rows with a unique class
        dataPartition = []
        for i in range(len(classNames)):
            dataPartition.append(self.partition_classes(classIndex, classNames[i], table))

        # distribute paritions roughly equally
        # TODO see note in log!!
        kPartitions = [[] for _ in range(k)]
        for partition in dataPartition:
            for i in range(len(partition)):
                kPartitions[i%k].append(partition[i])
        return kPartitions
 
def main():
    #mpg, cylinders, displacement, horsepower, weight, acceleration, model year, origin, car name       
    d = DataClassification()
    
    d.test_random_instances_step1()
    d.test_random_instances_step2()
    d.test_random_instances_step3()
    d.random_subsampling_accuracy_knn(10)
    #d.test_dan()


if __name__ == "__main__":
    main()
    


