#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""hw6.py:  Data mining assignment #6: Association Rule Mining
            Implements the apriori ARM algorithm and applies it to
            the titanic and mushroom datasets."""

__author__ = "Dan Collins and Miranda Myers"
import csv
import operator
from math import log
from tabulate import tabulate

class RuleFinder:

    def __init__(self, fileName, classIndex):
        """Constructor for RuleFinder class."""
        self.table = self.read_csv(fileName)
        self.attrNames = self.table.pop(0)
        self.ntotal = len(self.table)

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

    def match_rule_with_itemset(self, dataset, rule):
        nleft = 0
        for row in dataset:
            match = True
            for item in rule:
                if lhs[item] != row[item]: 
                    match = False
            if match == True:
                nleft += 1
        return nleft

    def calculate_nleft(self, dataset, rule):
        """Given a dataset and a rule data struct, returns nleft."""
        return self.match_rule_with_itemset(dataset, rule.lhs)

    def calculate_nright(self, dataset, rule):
        """Given a dataset and a rule data struct, returns nright."""
        return self.match_rule_with_itemset(dataset, rule.rhs)

    def calculate_nboth(self, dataset, rule):
        """Given a dataset and a rule data struct, returns nboth."""
        both = rule.lhs.update(rule.rhs)
        return self.match_rule_with_itemset(dataset, both)

    def calculate_confidence(self, nboth, nleft):
        """Given nboth and nleft, returns the confidence."""
        return float(nboth / nleft)

    def calculate_support(self, nboth, ntotal):
        """Given nboth and ntotal, returns the support."""
        return float(nboth / ntotal)

    def calculate_lift(self, dataset, rule):
        """Given a dataset and a rule in the form of [lhs, rhs],
           returns the lift."""

    def association_rule_mining(self):
        """."""
   
def main():
    """Creates objects to parse data files and finds / prints associated rules."""
    mushroom = RuleFinder('agaricus-lepiota.txt', 0)
    mushroom.test_rand_forest_ens('agaricus-lepiota.txt')

    titanic = RuleFinder('titanic.txt', 3)
    titanic.test_rand_forest_ens('titanic.txt')

if __name__ == "__main__":
    main()

