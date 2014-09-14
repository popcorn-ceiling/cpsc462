#!/usr/bin/env python

"""hw1.py: Data mining assignment #1. Data preparation exercises. \
           Reads in two files, auto-data.txt and auto-prices.txt. \
           Calculates statics on these files, joins them, cleans them, \
           and resolves missing values. Note there is some hard coding"""

__author__ = "Dan Collins and Miranda Myers"

import sys
import csv
import copy
import operator
from tabulate import tabulate

def read_csv(filename):
    """Reads in a csv file and returns a table as a list of lists (rows)."""
    the_file = open(filename)
    the_reader = csv.reader(the_file, dialect='excel')
    table = []
    for row in the_reader:
        if len(row) > 0:
            table.append(row)
    return table

def print_csv(table, fileName):
    """Prints a csv file from a table as a list of lists (rows)."""
    outFile = open(fileName, 'w')
    for row in table:
        for i in range(len(row) - 1):
            outFile.write(str(row[i]) + ',')
        outFile.write(str(row[-1]))
        outFile.write('\n')
    outFile.close()

def count_instances(dataset):
    """Returns the number instances in the dataset."""
    return len(dataset)

def find_duplicates(dataset, attributes):
    """Returns a list of the duplicates of a selected \
       attribute found in the dataset."""
    tempArray = []
    duplicates = [] 
    dupIndicies = []
    j = 0
    for row in dataset:
        list = []
        for i in attributes:
            list.append(row[i])
            if list in tempArray and list not in duplicates:
                duplicates.append(list)
                dupIndicies.append(j)
        tempArray.append(list)
        j += 1
    return duplicates, dupIndicies

def remove_duplicates(dataset, attributes):
    """Removes duplicates from a dataset based on a key passed in."""
    datasetNoDups = []
    duplicates, indexRemove = find_duplicates(dataset, attributes)
    i = 0
    for row in dataset:
        for item in indexRemove:
            if i != item:
                datasetNoDups.append(row)
        i += 1
    return datasetNoDups

def full_outer_join(dataset1, attributes1, dataset2, attributes2):
    """Performs a full outer join on two datasets
        Assumes attributes passed are in the same order for each dataset."""
    if len(attributes1) != len(attributes2):
        print 'Error, attribute size mismatch'
        exit()

    """First look map the second table to the first."""
    finalDataset = []
    for row1 in dataset1:
        joinedRow = False
        for row2 in dataset2:
            attributeMatch = True
            for i in range(len(attributes1)):
                if row1[attributes1[i]] != row2[attributes2[i]]:
                    attributeMatch = False
            if attributeMatch == True:
                finalDataset.append(row1 + [row2[-1]])
                joinedRow = True
        if joinedRow == False:
            finalDataset.append(row1 + ['NA'])

    """Now map the first to the second."""
    for row2 in dataset2:
        joinedRow = False
        for row1 in dataset1:
            attributeMatch = True
            for i in range(len(attributes1)):
                if row1[attributes1[i]] != row2[attributes2[i]]:
                    attributeMatch = False
            if attributeMatch == True:
                joinedRow = True
                if (row1 + [row2[-1]]) not in finalDataset:
                    finalDataset.append(row1 + row2[-1])
                    
        if joinedRow == False:
            newRow = ['NA', 'NA', 'NA', 'NA', 'NA', 'NA', \
                    row2[attributes2[0]], 'NA', row2[attributes2[1]], row2[-1]]
            finalDataset.append(newRow)

    return finalDataset

def get_column_as_floats(table, index):
    """Returns all non-null values in table for given index."""
    vals = []
    for row in table:
        if row[index] != "NA":
            vals.append(float(row[index]))
    return vals

def minimum(table, index):
    """Calculates the minimum value given a table and a column index."""
    return round(min(get_column_as_floats(table, index)), 2)

def maximum(table, index):
    """Calculates the maximum value given a table and a column index."""
    return round(max(get_column_as_floats(table, index)), 2)

def midpoint(table, index):
    """Calculates the midpoint value given a table and a column index."""
    minVal = minimum(table, index)
    maxVal = maximum(table, index)
    return round((minVal + maxVal) / 2.0, 2)
    
def average(table, index):
    """Calculates the average value given a table and a column index."""
    avgArray = get_column_as_floats(table, index)
    if len(avgArray) != 0:
        return round(sum(avgArray)/len(avgArray), 2)
    else:
        return 0
    
def median(table, index):
    """Calculates the median value given a table and a column index."""
    medianArray = get_column_as_floats(table, index)
    mid = len(medianArray) / 2
    if len(medianArray) % 2 == 0:
        median = (medianArray[mid - 1] + medianArray [mid]) / 2.0
    else: 
        median = medianArray[mid]
    return round(median, 2)

def calculate_summary_statistics(table, index, attributeTitle):
    """Calculates minimum, maximum, midpoint, average, and median \
       for a given attribute in a table."""
    summaryArray = [attributeTitle, \
                    minimum(table, index), \
                    maximum(table, index), \
                    midpoint(table, index), \
                    average(table, index), \
                    median(table, index)]
    return summaryArray

def resolve_by_remove(table):
    """Resolve missing values by removing instances with missing values."""
    resolvedTable = []
    for row in table:
        if 'NA' not in row:
            resolvedTable.append(row)
    return resolvedTable

def resolve_by_average(table):
    """Resolve by replacing missing values with their \
       corresponding attribute's average."""
    resolvedTable = copy.deepcopy(table)
    for row in resolvedTable:
        for index in range(len(row)):
            if row[index] == 'NA':
                row[index] = average(table, index)
    return resolvedTable

def build_yr_attr_table(table, year, attrVal, attrIndex):
    """Build table containing instances for given year and attribute."""
    newTable = []
    for row in table:
        if row[6] == year and attrVal == row[attrIndex]:
            newTable.append(row)
    return newTable

def resolve_by_meaningful_avg(table, attribute):
    """Resolve by placing missing values with average \
       of year and some given attribute."""
    resolvedTable = copy.deepcopy(table)
    for row in resolvedTable:
        for index in range(len(row)):
            if row[index] == 'NA':
                newTable = build_yr_attr_table(table, row[6], row[attribute], attribute)
                row[index] = average(newTable, index)
    return resolvedTable

def print_inst_dups(dataset, header, attributes):
    """Calculates and prints the number of instances \
       and duplicates for a given dataset."""
    datasetInstDups = []
    instances = count_instances(dataset)
    duplicates, throwAway = find_duplicates(dataset, attributes)
    datasetInstDups = [['No. of instaces: ' + str(instances)], \
                       ['Duplicates: ']]
    for item in duplicates:
        itemAsList = [str(item)]
        datasetInstDups.append(itemAsList)
    
    print tabulate(datasetInstDups, header, tablefmt='simple') 
    print

def print_summary_stats(dataset, header):
    """Calculates and prints the summary statistics for a given dataset."""
    summStat = []
    for i in range(len(header)):
        """Not dealing with categorical attributes at the moment!"""
        if header[i] != 'car name':
            summData = calculate_summary_statistics(dataset, i, header[i])
            summStat.append(summData)
    
    hdrSummStat = ['attribute', 'min', 'max', 'mid', 'avg', 'med']
    print '----------------------------------------'
    print 'Summary Stats:'
    print tabulate(summStat, hdrSummStat, tablefmt='rst', numalign='right')
    print

def main():
    """Reads in auto-mpg.txt and auto-prices.txt, \
       calculates statistics on them, joins them, \
       and cleans them in various ways. Generates auto-data.txt \
       (full join on data sets)."""

    fileNames = ['auto-mpg.txt','auto-prices.txt', 'auto-mpg-nodups.txt', \
                 'auto-prices-nodups.txt', 'auto-mpg-clean.txt', \
                 'auto-prices-clean.txt', 'auto-data.txt']

    attrMpg = [6, 8]
    attrPrice = [1, 0]
    attributes = [attrMpg, attrPrice]

    # mpg file
    dataMpg = []
    dataMpg = read_csv(fileNames[0])
    print '--------------------'
    print_inst_dups(dataMpg, [fileNames[0]], attributes[0])
    
    # price file
    dataPrice = []
    dataPrice = read_csv(fileNames[1])
    print '---------------------'
    print_inst_dups(dataPrice, [fileNames[1]], attributes[1])

    # remove dups from mpg and price, save as new files
    # done manually outside of file
    # we also manually fixed some typos here (toyoto -> toyota)
    dataMpgNoDups = []
    dataMpgNoDups = read_csv(fileNames[2])
    print '--------------------'
    print_inst_dups(dataMpgNoDups, [fileNames[2]], attributes[0])
    
    dataPriceNoDups = []
    dataPriceNoDups = read_csv(fileNames[3])
    print '---------------------'
    print_inst_dups(dataPriceNoDups, [fileNames[3]], attributes[1])

    # read in clean data. for mpg, read in nodups.txt 
    # and replace instances missing
    # mpg data with the years average for that attribute
    dataMpgClean = read_csv(fileNames[4])
    print '--------------------'
    print_inst_dups(dataMpgClean, [fileNames[4]], attributes[0])
    dataPriceClean = read_csv(fileNames[5])
    print '--------------------'
    print_inst_dups(dataPriceClean, [fileNames[5]], attributes[1])

    # full outer join previous datasets, save to file
    dataFullJoin = []
    dataFullJoin = full_outer_join(dataMpgClean, attrMpg, \
                                   dataPriceClean, attrPrice)
    dataFullJoin.sort(key=operator.itemgetter(6,8))

    # resolve cases in which there is no mpg data
    resolvedTable = copy.deepcopy(dataFullJoin)
    for row in resolvedTable:
        for index in range(len(row)):
            if row[index] == 'NA' and row[index] != row[-1]:
                newTable = build_yr_attr_table(dataFullJoin, row[6], \
                                               row[6], 6)
                row[index] = median(newTable, index)
    dataFullJoin = resolvedTable
    print_csv(dataFullJoin, fileNames[-1])
    
    hdrStats = ['mpg', 'cylinders', 'displacement', 'horsepower', \
                'weight', 'acceleration', 'model year', 'origin', \
                'car name', 'msrp'] 
    hdrCombined = ['combined data (saved as auto-data.txt)']
    print '----------------------------------------'
    print_inst_dups(dataFullJoin, [hdrCombined[0]], attributes[0])
    print_summary_stats(dataFullJoin, hdrStats)

    # resolve missing values by various methods
    # 7 is index of country of origin
    # which taken with year will be used to avg
    # resolve by removing instances with NA
    dataJoinClean = resolve_by_remove(dataFullJoin)
    hdrCombined = ['combined table (rows w/ missing values removed)']
    print '-------------------------------------------------'
    print_inst_dups(dataJoinClean, [hdrCombined[0]], attributes[0])
    print_summary_stats(dataJoinClean, hdrStats)
    
    # resolve by attribute avg
    dataJoinClean = resolve_by_average(dataFullJoin)
    hdrCombined = ['combined table (missing values averaged by attribute)']
    print '-------------------------------------------------------'
    print_inst_dups(dataJoinClean, [hdrCombined[0]], attributes[0])
    print_summary_stats(dataJoinClean, hdrStats)

    # resolve by meanigful avg
    dataJoinClean = resolve_by_meaningful_avg(dataFullJoin, 7)
    hdrCombined = ['combined table (missing values meaningfully averaged)']
    print '-------------------------------------------------------'
    print_inst_dups(dataJoinClean, [hdrCombined[0]], attributes[0])
    print_summary_stats(dataJoinClean, hdrStats)
    print tabulate(dataJoinClean)
    print_csv(dataJoinClean, fileNames[-1])

if __name__ == "__main__":
    main()

# bugs to fix - cleaning
