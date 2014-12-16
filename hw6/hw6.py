#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""hw6.py:  Data mining assignment #6: Association Rule Mining
            Implements the apriori ARM algorithm and applies it to
            the titanic and mushroom datasets."""

__author__ = "Dan Collins and Miranda Myers"
import csv
import operator
import copy
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
        """Returns the number of times a given rule occurs in a dataset.
           note that we deal with [index, attrVal]."""
        count = 0
        for row in self.table:
            match = True
            for item in ruleSubset:
                index = item[0] # 0 is index of column
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
        both = rule.rhs + rule.lhs
        return self.match_rule_with_itemset(both)

    def calculate_confidence(self, nboth, nleft):
        """Given nboth and nleft, returns the confidence."""
        return nboth / (nleft*1.0)

    def calculate_support(self, nboth, ntotal):
        """Given nboth and ntotal, returns the support."""
        return nboth / (ntotal * 1.0)

    def calculate_lift(self, conf, nRight):
        """Given a dataset and a rule in the form of [lhs, rhs],
           returns the lift."""
        return ((self.ntotal * conf) / (nRight * 1.0))

    def create_c1(self, indices):
        """Creates c1 (all candidate itemsets of size 1) for apriori."""
        c1 = []
        for transaction in self.table:
            for index, item in enumerate(transaction):
                if index not in indices:
                    continue
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
        
    def apriori(self, minSup, indices):
        """Generates all supported itemsets."""
        c1 = self.create_c1(indices)
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

    def generate_rules(self, itemsets, minConf):
        """Finds confident rules from a supported itemset (i.e. L3)."""
        ruleObj = Rule()

        for lk in itemsets:
            k = len(lk[0])
            lhsList, rhsList = [], []
            confList, supList, liftList = [], [], []

            # loop through all members of lk
            for item in lk:
                rhsBL = []
                # generate all RHS <= k
                for i in range(1, k):
                    rhsGen = [list(x) for x in combinations(item, i)]
                
                    # find all associated LHS rules and confidence of L->R
                    for rhs in rhsGen:
                        # check if we know this rhs to be garbage
                        if (rhs in rhsBL):
                            continue

                        ruleObj.rhs = rhs
                        ruleObj.lhs = [x for x in item if x not in rhs]

                        nleft = self.calculate_nleft(ruleObj)
                        nboth = self.calculate_nboth(ruleObj)
                        conf = self.calculate_confidence(nboth, nleft)

                        # if we'd let this guy watch our kids, add to whitelist
                        if (conf >= minConf):
                            rhsList.append(rhs)
                            lhsList.append(ruleObj.lhs)
                            confList.append(conf)

                            sup = self.calculate_support(nboth, self.ntotal)
                            supList.append(sup)

                            nright = self.calculate_nright(ruleObj)
                            lift = self.calculate_lift(conf, nright)
                            liftList.append(lift)
                        # else add to blacklist
                        else:
                            rhsBL.append(rhs)

        return rhsList, lhsList, confList, supList, liftList

    def ar_print(self, rhs, lhs, conf, sup, lift):
        """Pretty prints a table containing association rules 
           and their measures of interestingness."""
        headers = ['association rule', 'support', 'confidence', 'lift']
        printTable = []
        for i in range(len(rhs)):
            arString = str(i)
            for item in lhs[i]:
                attrName = str(self.attrNames[item[0]]) + '='
                attrVal = item[1]
                arString += ' ' + attrName + attrVal
            arString += '   =>   '
            for item in rhs[i]:
                attrName = str(self.attrNames[item[0]]) + '='
                attrVal = item[1]
                arString += ' ' + attrName + attrVal
            row = [arString, \
                    str(round(sup[i], 5)), \
                    str(round(conf[i], 5)), \
                    str(round(lift[i], 5))]
            printTable.append(row)
        print (tabulate(printTable, headers, tablefmt="rst"))

def main():
    """Creates objects to parse data files and finds associated rules."""
    print 'hold tight, this may take a minute...'

    minSup = 0.05
    minConf = 0.8

    # mushroom dataset
    mushroom = RuleFinder('agaricus-lepiota.txt')
    indices = [0,3,6,12,17,22]
    mushItemsets = mushroom.apriori(minSup, indices)
    print 'done with mushroom itemsets...'
    mRHS, mLHS, mCONF, mSUP, mLIFT = \
        mushroom.generate_rules(mushItemsets, minConf)
    print 'done with mushroom association rules...'
    mushroom.ar_print(mRHS, mLHS, mCONF, mSUP, mLIFT)


    # titanic dataset
    titanic = RuleFinder('titanic.txt')
    indices = [x for x in range(len(titanic.table[0]))]
    titanicItemsets = titanic.apriori(minSup, indices)
    print 'done with titanic itemsets'
    tRHS, tLHS, tCONF, tSUP, tLIFT = \
        titanic.generate_rules(titanicItemsets, minConf)
    print 'done with titanic association rules...'
    titanic.ar_print(tRHS, tLHS, tCONF, tSUP, tLIFT)


if __name__ == "__main__":
    main()

