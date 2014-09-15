
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

def calculate_frequencies(xs):
    """Returns a unique, sorted list of values in xs and occurrence \
        counts for each value."""
    ys = sorted(xs)
    values, counts = [], []
    for y in ys:
        if y not in values:
            values.append(y)
            counts.append(1)
        else:
            counts[-1] += 1
    return values, counts        
        
def create_frequency_diagram(table, index, title, xlabel, ylabel, outfile):
    """Creates a frequency diagram for a given categorical attribute."""
    xs = get_column_as_floats(table, index)
    values, counts = calculate_frequencies(xs)
            
    #Reset the figure
    pyplot.figure()
    
    #Generate histogram
    pyplot.bar(values, counts, align='center')
    pyplot.grid(True)
    pyplot.title(title)
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.xticks(values)
    
    pyplot.savefig(outfile)

def create_all_freq_diagrams():
    """Creates frequency diagrams of all categorical attributes of auto-data.txt dataset."""
    table = read_csv('auto-data.txt')
    
    #Create cylinder diagram
    create_frequency_diagram(table, 1, 'Total Number of Cars by Number of Cylinders', \
        'Cylinders', 'Count', 'step-1-cylinders.pdf')
    
    #Create model year diagram
    create_frequency_diagram(table, 6, 'Total Number of Cars by Year', 'Year', 'Count', \
        'step-1-modelyear.pdf')
    
    #Create origin diagram
    create_frequency_diagram(table, 7, 'Total Number of Cars by Origin', 'Origin', \
        'Count', 'step-1-origin.pdf')
        
def create_pie_chart(table, index, title, outfile):
    """Creates a pie chart for a given categorical attribute."""
    xs = get_column_as_floats(table, index)
    values, counts = calculate_frequencies(xs)
    values = map(int, values)
    
    pyplot.figure()
    pyplot.pie(counts, labels=values, colors=('#00FFFF', '#0000FF', '#8A2BE2', \
        '#7FFF00', '#FF7F50', '#FF1493', '#DA70D6', '#FFFF00', '#87CEEB', '#3CB371'), \
        autopct='%1.1f%%')
    pyplot.title(title)
    pyplot.axis('equal')
    
    pyplot.savefig(outfile)
    
    
def create_all_pie_charts():
    """Creates pie charts of all categorical attributes of auto-data.txt dataset."""
    table = read_csv('auto-data.txt')
    
    #Create cylinder pie chart
    create_pie_chart(table, 1, 'Total Number of Cars by Number of Cylinders\n', 'step-2-cylinders.pdf')
    
    #Create model year pie chart
    create_pie_chart(table, 6, 'Total Number of Cars by Year\n', 'step-2-modelyear.pdf')
    
    #Create origin pie chart
    create_pie_chart(table, 7, 'Total Number of Cars by Origin', 'step-2-origin.pdf')



def main():
    create_all_freq_diagrams()
    create_all_pie_charts()


main()
    






