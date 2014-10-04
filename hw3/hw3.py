#!/usr/bin/env python

"""hw3.py:  Data mining assignment #3: Data classification."""

__author__ = "Dan Collins and Miranda Myers"

import numpy 
import csv
import operator

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

    def discretize_mpg_DoE(self, index):
        """Converts mpg into a categorical attribute using US \
           Department of Energy ratings."""
        xs = self.get_column_as_floats(self.__table, index)
        counts = [0 for i in range(10)]
        ratings = [i+1 for i in range(10)]
    
        #Creates frequency counts based on the ratings
        for i in range(len(xs)):
            if xs[i] <= 13:
                counts[0] += 1
            if xs[i] == 14:
                counts[1] += 1
            if xs[i] >= 15 and xs[i] <= 16:
                counts[2] += 1
            if xs[i] >= 17 and xs[i] <= 19:
                counts[3] += 1
            if xs[i] >= 20 and xs[i] <= 23:
                counts[4] += 1
            if xs[i] >= 24 and xs[i] <= 26:
                counts[5] += 1
            if xs[i] >= 27 and xs[i] <= 30:
                counts[6] += 1
            if xs[i] >= 31 and xs[i] <= 36:
                counts[7] += 1
            if xs[i] >= 37 and xs[i] <= 44:
                counts[8] += 1
            if xs[i] >= 45:
                counts[9] += 1

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
    
def main():
    """FIXME."""


if __name__ == "__main__":
    main()
    
