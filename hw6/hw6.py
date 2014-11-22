#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""hw6.py:  Data mining assignment #6: Association Rule Mining
            Implements the apriori ARM algorithm and applies it to
            the titanic and mushroom datasets."""

__author__ = "Dan Collins and Miranda Myers"
import copy
import csv
import random
import operator
import numpy
import numpy.ma as ma
from math import log
from tabulate import tabulate

class RuleFinder:

    def __init__(self, fileName, classIndex):
        """Constructor for RuleFinder class."""
        self.table = self.read_csv(fileName)
        self.attrNames = self.table.pop(0)
        self.uniqueClasses = []
        for row in self.table:
            if row[self.classIndex] not in self.uniqueClasses:
                self.uniqueClasses.append(row[self.classIndex])
        self.uniqueClasses.sort()

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

    def calculate_confidence(self, dataset, rule):
        """Given a dataset and a rule in the form of [lhs, rhs],
           returns the confidence."""

    def calculate_support(self, dataset, rule):
        """Given a dataset and a rule in the form of [lhs, rhs],
           returns the support."""

    def calculate_lift(self, dataset, rule):
        """Given a dataset and a rule in the form of [lhs, rhs],
           returns the lift."""

    def calculate_cross_entropy(self, dataset, rule):
        """Given a dataset and a rule in the form of [lhs, rhs],
           returns the cross entropy."""

    def calculate_j_measure(self, dataset, rule):
        """Given a dataset and a rule in the form of [lhs, rhs], 
           returns the J-Measure."""
   
def main():
    """Creates objects to parse data files and finds / prints associated rules."""
    mushroom = RuleFinder('agaricus-lepiota.txt', 0)
    mushroom.test_rand_forest_ens('agaricus-lepiota.txt')

    titanic = RuleFinder('titanic.txt', 3)
    titanic.test_rand_forest_ens('titanic.txt')

if __name__ == "__main__":
    main()

