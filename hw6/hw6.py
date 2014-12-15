#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""hw6.py:  Data mining assignment #6: Association Rule Mining
            Implements the apriori ARM algorithm and applies it to
            the titanic and mushroom datasets."""

__author__ = "Dan Collins and Miranda Myers"
import csv
import operator
import copy
from rule import Rule
from math import log
from tabulate import tabulate
from itertools import combinations

class Rule:
    """Holds rules in an easy to read manner."""

    def __init__(self):
        self.lhs = []
        self.rhs = []

class RuleFinder:
    """Contains apriori algorithm to generate supported itemsets
       and find rules from them."""

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
        """Returns the number of times a given rule occurs in a dataset."""
        count = 0
        for row in self.table:
            match = True
            for item in ruleSubset:
                index = item[0]
                if item[1] != row[index]: 
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
        both = rule.lhs.update(rule.rhs) #TODO won't work with lists
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
        """Creates c1 (all candidate itemsets of size 1) for apriori."""
        c1 = []
        for transaction in self.table:
            for index, item in enumerate(transaction):
                if [index, item] not in c1:
                    c1.append([index, item])
        c1.sort(key=operator.itemgetter(0,1))
        cfinal = []
        for item in c1:
            cfinal.append([item])
        return cfinal

    def is_supported(self, candidate, minSup):
        """Returns true if a rule is supported in a dataset."""
        count = self.match_rule_with_itemset(candidate) * 1.0
        support = count / self.ntotal
        return support >= minSup      
 
    def perform_union(self, list1, list2):
        """Perform a union on two lists with final list sorted."""
        uList = copy.deepcopy(list1)
        for item in list2:
            if item not in list1:
                uList.append(item)
        uList.sort(key = operator.itemgetter(0,1))
        return uList

    def apriori_gen(self, lk_1):
        """Creates ck from lk-1."""
        ck = []
        # Join step
        for i in range(len(lk_1)):
            curItemset = lk_1[i]
            # Iterate through each itemset that follows the current itemset
            for itemset in lk_1[i+1:]:
                # Check if all but the last items are the same
                if curItemset[:-1] == itemset[:-1]:
                    # Perform union, sort, and append to ck
                    union = self.perform_union(curItemset, itemset)
                    ck.append(union)
        
        # Prune step    
        pruned_ck = []
        for itemset in ck:
            # Check if each subset is member of lk_1
            add = True 
            for i in range(len(itemset)):
                subset = itemset[:i] + itemset[i+1:]
                if subset != [] and subset not in lk_1:  
                    add = False
                    break
            if add == True:
                pruned_ck.append(itemset)
        
        return pruned_ck
    
    def create_lk(self, ck, minSup):
        """Takes a candidate itemset and returns only the supported items of it."""
        lk = []
        for item in ck:
            if self.is_supported(item, minSup):
                lk.append(item)
        return lk
        
    def apriori(self, minSup):
        """Generates Ck from Lk_1 based on a minimum support value."""
        c1 = self.create_c1()
        lk_1 = self.create_lk(c1, minSup)
         
        k = 2
        L = []
        ck = []
        while len(lk_1) != 0:
            # Creates ck from lk-1
            ck = self.apriori_gen(lk_1)

            # Creates lk by pruning unsupported itemsets
            lk = self.create_lk(ck, minSup)
            if lk == []:
                return L        

            L.append(lk)
            k += 1
            lk_1 = lk

    def generate_size_k_RHS(self, itemsets, minConf):
        """Finds confident rules from a supported itemset (i.e. L3)."""
        ruleObj = Rule()

        # implement short cicuits TODO
        # keep track of RHS not to check TODO
        for itemset in itemsets:
            masterRHS = []
            masterLHS = []
            k = len(itemsets[0])
            for i in range(1, k):
                rhsList = [list(x) for x in combinations(itemset, i)]
                lhsList = []
                for rhs in rhsList:
                    lhs = [x for x in itemset if x not in rhs]
                    lhsList.append(lhs)
                masterRHS = masterRHS + rhsList
                masterLHS = masterLHS + lhsList

        # put in rule data struct TODO
            print 'k', k
            print 'rhs', masterRHS
            print 'lhs', masterLHS

def main():
    """Creates objects to parse data files and finds / prints associated rules."""
    headers = ['association rule', 'support', 'confidence', 'lift']
    minSup = 0.6
    minConf = 0.8

    #mushroom = RuleFinder('agaricus-lepiota.txt')
    #mushItemsets = mushroom.apriori(minSup)
    #mushRules = mushroom.generateRules(mushItemsets, minConf)

    titanic = RuleFinder('titanic.txt')
    #titanicItemsets = titanic.apriori(minSup)
    #titanicRules = titanic.generateRules(titanicItemsets, minConf)

    testitem = [['a','b','c'], ['d','e','f']]
    titanic.generate_size_k_RHS(testitem, minConf)

if __name__ == "__main__":
    main()

