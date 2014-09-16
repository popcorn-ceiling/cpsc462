
import matplotlib.pyplot as pyplot
import csv

class DataVisualization:

    def __init__(self):
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

    def calculate_frequencies(self, xs):
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
        
    def create_frequency_diagram(self, table, index, title, xlabel, ylabel, outfile):
        """Creates a frequency diagram for a given categorical attribute."""
        xs = self.get_column_as_floats(table, index)
        values, counts = self.calculate_frequencies(xs)
            
        #Resets the figure
        pyplot.figure()
    
        #Generates frequency diagram
        pyplot.bar(values, counts, align='center')
        pyplot.grid(True)
        pyplot.title(title)
        pyplot.xlabel(xlabel)
        pyplot.ylabel(ylabel)
        pyplot.xticks(values)
    
        pyplot.savefig(outfile)

    def create_all_freq_diagrams(self):
        """Creates frequency diagrams of all categorical attributes of auto-data.txt dataset."""
    
        #Creates cylinder diagram
        self.create_frequency_diagram(self.__table, 1, 'Total Number of Cars by Number of Cylinders', \
            'Cylinders', 'Count', 'step-1-cylinders.pdf')
    
        #Creates model year diagram
        self.create_frequency_diagram(self.__table, 6, 'Total Number of Cars by Year', 'Year', 'Count', \
            'step-1-modelyear.pdf')
    
        #Creates origin diagram
        self.create_frequency_diagram(self.__table, 7, 'Total Number of Cars by Origin', 'Origin', \
            'Count', 'step-1-origin.pdf')
        
    def create_pie_chart(self, table, index, title, outfile):
        """Creates a pie chart for a given categorical attribute."""
        xs = self.get_column_as_floats(table, index)
        values, counts = self.calculate_frequencies(xs)
        values = map(int, values)
    
        pyplot.figure()
        pyplot.pie(counts, labels=values, colors=('#00FFFF', '#0000FF', '#8A2BE2', \
            '#7FFF00', '#FF7F50', '#FF1493', '#DA70D6', '#FFFF00', '#87CEEB', '#3CB371'), \
            autopct='%1.1f%%')
        pyplot.title(title)
        pyplot.axis('equal')
    
        pyplot.savefig(outfile)
    
    def create_all_pie_charts(self):
        """Creates pie charts of all categorical attributes of auto-data.txt dataset."""
    
        #Creates cylinder pie chart
        self.create_pie_chart(self.__table, 1, 'Total Number of Cars by Number of Cylinders\n', 'step-2-cylinders.pdf')
    
        #Creates model year pie chart
        self.create_pie_chart(self.__table, 6, 'Total Number of Cars by Year\n', 'step-2-modelyear.pdf')
    
        #Creates origin pie chart
        self.create_pie_chart(self.__table, 7, 'Total Number of Cars by Origin', 'step-2-origin.pdf')

    def create_dot_chart(self, table, index, title, xlabel, outfile):
        xs = self.get_column_as_floats(table, index)
        values, counts = self.calculate_frequencies(xs)
     
        #Initializes all ys to 1
        ys = [1 for item in xs]  

        pyplot.figure()
        pyplot.title(title)
        pyplot.xlabel(xlabel)
        pyplot.plot(xs, ys, alpha=0.2, marker='.', markersize=15, linestyle='None')
        pyplot.gca().get_yaxis().set_visible(False)
        pyplot.savefig(outfile)
    

    #mpg, cylinders, displacement, horsepower, weight, acceleration, model year, origin, and car name
    def create_all_dot_charts(self):
    
        #Creates mpg dot chart
        self.create_dot_chart(self.__table, 0, 'Miles Per Gallon of All Cars', 'MPG', 'step-3-mpg.pdf')
    
        #Creates displacement dot chart
        self.create_dot_chart(self.__table, 2, 'Displacement of All Cars', 'Displacement', 'step-3-displacement.pdf')
    
        #Creates horesepower dot chart
        self.create_dot_chart(self.__table, 3, 'Horsepower of All Cars', 'Horsepower', 'step-3-horsepower.pdf')
    
        #Creates weight dot chart
        self.create_dot_chart(self.__table, 4, 'Weight of All Cars', 'Weight', 'step-3-weight.pdf')
    
        #Creates acceleration dot chart
        self.create_dot_chart(self.__table, 5, 'Acceleration of All Cars', 'Acceleration', 'step-3-acceleration.pdf')

        #Creates msrp dot chart
        self.create_dot_chart(self.__table, 9, 'MSRP of All Cars', 'MSRP', 'step-3-msrp.pdf')

    def transfor_mpg_DoE(self, index):
        """Converts mpg into a categorical attribute using US Department of Energy ratings."""
        xs = self.get_column_as_floats(self.__table, index)
        counts = [0 for i in range(10)]
        ratings = [i+1 for i in range(10)]
    
        #Create frequency counts based on the ratings
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
                    
        #Resets the figure
        pyplot.figure()
    
        #Generates frequency diagram
        pyplot.bar(ratings, counts, align='center')
        pyplot.grid(True)
        pyplot.title('hi')
        pyplot.xticks(ratings)
    
        pyplot.savefig('somename.pdf')
            
        #This function probably isn't done

def main():
    visualizationObject = DataVisualization()
    
    visualizationObject.create_all_freq_diagrams()
    visualizationObject.create_all_pie_charts()
    visualizationObject.create_all_dot_charts()
    visualizationObject.transfor_mpg_DoE(0)


main()
    






