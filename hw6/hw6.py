#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""hw6.py:  Data mining assignment #6: Association Rule Mining
            Implements the apriori ARM algorithm and applies it to
            the titanic and mushroom datasets."""

__author__ = "Dan Collins and Miranda Myers"
import csv
import operator
from rule import Rule
from math import log
from tabulate import tabulate

class RuleFinder:

    def __init__(self, fileName):
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

    def match_rule_with_itemset(self, ruleSubset):
        count = 0
        for row in self.table:
            match = True
            for item in ruleSubset:
                if ruleSubset[item] != row[item]: 
                    match = False
            if match == True:
                count += 1
        return count

    def calculate_nleft(self, rule):
        """Given a dataset and a rule data struct, returns nleft."""
        return self.match_rule_with_itemset(rule.lhs)

    def calculate_nright(self, rule):
        """Given a dataset and a rule data struct, returns nright."""
        return self.match_rule_with_itemset(rule.rhs)

    def calculate_nboth(self, rule):
        """Given a dataset and a rule data struct, returns nboth."""
        both = rule.lhs.update(rule.rhs)
        return self.match_rule_with_itemset(both)

    def calculate_confidence(self, nboth, nleft):
        """Given nboth and nleft, returns the confidence."""
        return float(nboth / nleft)

    def calculate_support(self, nboth, ntotal):
        """Given nboth and ntotal, returns the support."""
        return float(nboth / ntotal)

    def calculate_lift(self, nLeft, nRight, nBoth, support):
        """Given a dataset and a rule in the form of [lhs, rhs],
           returns the lift."""
        lUnionR = (nLeft + nRight) - nBoth        
        lift = lUnionR / (nLeft * support)
        return lift

    def create_c1(self):
        """Creates c1 for apriori."""
        c1 = {}
        for transaction in self.table:
            for item in transaction:
                index = transaction.index(item)
                if index not in c1:
                    c1.update({index : [item]})
                else:
                    valList = c1[index]
                    c1.update({index : valList.append(item)})
        return c1

    def association_rule_mining(self):
        """."""
        pass

   
def main():
    """Creates objects to parse data files and finds / prints associated rules."""
    mushroom = RuleFinder('agaricus-lepiota.txt')
    mushroom.association_rule_mining()

    titanic = RuleFinder('titanic.txt')
    titanic.association_rule_mining()

if __name__ == "__main__":
    main()

