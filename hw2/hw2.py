
import matplotlib.pyplot as pyplot
import csv


def read_csv(filename):
    """Reads in a csv file and returns a table as a list of lists (rows)."""
    the_file = open(filename)
    the_reader = csv.reader(the_file, dialect='excel')
    table = []
    for row in the_reader:
        if len(row) > 0:
            table.append(row)
    return table


def get_column_as_floats(table, index):
    """Returns all non-null values in table for given index."""
    vals = []
    for row in table:
        if row[index] != "NA":
            vals.append(float(row[index]))
    return vals
    

def frequencies(xs):
    """Returns a unique, sorted list of values in xs and occurrence \
        counts for each value"""
    ys = sorted(xs)
    values, counts = [], []
    for y in ys:
        if y not in values:
            values.append(y)
            counts.append(1)
        else:
            counts[-1] += 1
    return values, counts
        
        
def create_frequency_diagram(table, index):
    """Creates a frequency diagram for a given categorical attribute"""
    xs = get_column_as_floats(table, index)
    values, counts = frequencies(xs)
    
    print 'VALUES: ', values, 'COUNTS: ', counts
        
    #Reset the figure
    pyplot.figure()
    
    #Generate histogram
    #pyplot.hist(values, bins=counts)
    #pyplot.show()
    

#def display_all_freq_diagrams():


def main():

    #Create frequency diagram for origin
    table = read_csv('auto-data.txt')
    create_frequency_diagram(table, 7)



main()
    






