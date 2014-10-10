#!/usr/bin/env python

import math
import operator


"""hw3.py:  Data mining assignment #3: Data classification."""

__author__ = "Dan Collins and Miranda Myers"

import numpy 
import csv
import operator
import random

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
                   
        if y < 14:
            rating = 1
        elif y == 14:
            rating = 2
        elif y > 14 and y <= 16:
            rating = 3
        elif y > 16 and y <= 19:
            rating = 4
        elif y > 19 and y <= 23:
            rating = 5
        elif y > 23 and y <= 26:
            rating = 6
        elif y > 26 and y <= 30:
            rating = 7
        elif y > 30 and y <= 36:
            rating = 8
        elif y > 36 and y <= 44:
            rating = 9
        elif y > 44:
            rating = 10
        #Implement error checking else statement
        #Ask bowers about edges of bins
                
        return rating
        
    #def print_mpg_classification(self):
        

    def test_random_instances(self):
        print '==========================================='
        print 'STEP 1: Linear Regression MPG Classifier'
        print '==========================================='
        
        weights = self.get_column_as_floats(self.__table, 4)
                    
        testValues = []
        printIndices = []
        for i in range(5):
            rand_i = random.randint(0, len(weights))
            val = weights[rand_i]
            
            printIndices.append(rand_i)
            testValues.append(val)
                        
        for i in range(len(testValues)):
            classification = self.classify_mpg_lr(self.__table, 4, testValues[i])
            instance = self.__table[printIndices[i]]
            actual = float(instance[0])
            actual = self.classify_mpg_DoE(actual)
            print '    instance: ', instance
            print '    class: ', classification, 'actual: ', actual   
            
    def normalize(self, xs):
        minval = min(xs)
        maxval = max(xs)
        maxmin = (maxval - minval) * 1.0
        return [(x - minval) / maxmin for x in xs] 
            
    def calculate_euclidean_distance(self, row, instance, indices):
        distance_sum = 0.0
        for i in indices:
            distance_sum += (float(row[i]) - float(instance[i])) ** 2
        
        return math.sqrt(distance_sum)
    
    
    def k_nn_classifier(self, trainingSet, indices, instance, k, class_index):
        row_distances = []
        
        columns = []
        for i in indices:
            column = self.get_column_as_floats(self.__table, i)
            columns.append(column)
        print 'lencolums', len(columns), '\n'
        
   
        normalizedTrainingSet = []
        for j in range(len(trainingSet)):
            newRow = trainingSet[j][:]
            for i in range(len(columns)):
                normalizedColumn = self.normalize(columns[i])
                newRow[indices[i]] = normalizedColumn[j]
                normalizedTrainingSet.append(newRow)
        
        for row in normalizedTrainingSet:
            print row
        
        for row in normalizedTrainingSet:
            row_distances.append([self.calculate_euclidean_distance(row, instance, indices), row])
        
        row_distances.sort(key = operator.itemgetter(0))
        label = self.select_class_label(row_distances[0:k-1], class_index)
    
        return label
        
    
    def select_class_label(self, closest_k, class_index):
        ''' Select the class label for the nearest k neighbors. '''
        # Assign points to the nearest k neighbors
            # Points start at 1 for the farthest away and increment by one up to the 
            # nearest neighbor
        labels = []
        points = []
        for i in range(len(closest_k) - 1, 0, -1):
            labels.append(closest_k[i][class_index])
            points.append(i)
        
        # Create a dictionary of the labels with corresponding total points 
        pointLabelDict = {}
        for i in range(len(labels)):
            if labels[i] not in pointLabelDict.keys():
                pointLabelDict.update({labels[i] : points[i]})
            else:
                pointLabelDict[labels[i]] += points[i]
                
        print pointLabelDict
                        
        maxPoints = max(pointLabelDict.values())
        maxKeys = [x for x,y in pointLabelDict.items() if y ==maxPoints]
        
        #Find key(s) with the max total points
        print maxKeys
        
    #mpg, cylinders, displacement, horsepower, weight, acceleration, model year, origin, car name       
    def test(self):
        trainingSet = self.__table[0:20]
        indices = [1, 4, 5]
        instance = self.__table[30]
        k = 5
        class_index = 0
        self.k_nn_classifier(trainingSet, indices, instance, k, class_index)
    
    
    def discretize_weight_nhtsa(self, weights):
        categoricalWeights = []
        for item in weights:
            if item < 2000:
                categoricalWeights.append(1)
            elif item >= 2000 and item < 2500:
                categoricalWeights.append(2)
            elif item >= 2500 and item < 3000:
                categoricalWeights.append(3)
            elif item >= 3000 and item < 3500:
                categoricalWeights.append(4)
            elif item >= 3500:
                categoricalWeights.append(5)
            else:
                print 'error in discretize_weight'
                exit(-1)
    
            return categoricalWeights

    def gaussian(self, x, mean, sdev):
        first, second = 0, 0
        if sdev > 0:
            first = 1 / (mat.sqrt(2 * math.pi) * sdev)
            second = math.e ** (-((x - mean) ** 2) / (2 * (sdev ** 2)))
        return first * second

def main():
    d = DataClassification()
    
    #d.test_random_instances()
    d.test()


if __name__ == "__main__":
    main()
    









