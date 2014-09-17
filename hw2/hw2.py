
import matplotlib.pyplot as pyplot
import csv
import operator

class DataVisualization:

    def __init__(self, ):
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
        pyplot.close()

    def create_all_freq_diagrams(self):
        """Creates frequency diagrams of all categorical attributes of auto-data.txt dataset."""
        self.create_frequency_diagram(self.__table, 1, 'Total Number of Cars by Number of Cylinders', \
            'Cylinders', 'Count', 'step-1-cylinders.pdf')
        self.create_frequency_diagram(self.__table, 6, 'Total Number of Cars by Year', 'Year', 'Count', \
            'step-1-modelyear.pdf')
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
        pyplot.close()
    
    def create_all_pie_charts(self):
        """Creates pie charts of all categorical attributes of auto-data.txt dataset."""
        self.create_pie_chart(self.__table, 1, 'Total Number of Cars by Number of Cylinders\n', 'step-2-cylinders.pdf')
        self.create_pie_chart(self.__table, 6, 'Total Number of Cars by Year\n', 'step-2-modelyear.pdf')
        self.create_pie_chart(self.__table, 7, 'Total Number of Cars by Origin', 'step-2-origin.pdf')

    def create_dot_chart(self, table, index, title, xlabel, outfile):
        xs = self.get_column_as_floats(table, index)
        values, counts = self.calculate_frequencies(xs)
     
        #Initializes all ys to 1
        ys = [1 for item in xs]  

        pyplot.figure()
        pyplot.title(title)
        pyplot.xlabel(xlabel)
        pyplot.plot(xs, ys, alpha=0.2, marker='.', markersize=16, linestyle='None')
        pyplot.gca().get_yaxis().set_visible(False)
        
        pyplot.savefig(outfile)
        pyplot.close()

    def create_all_dot_charts(self):
        self.create_dot_chart(self.__table, 0, 'Miles Per Gallon of All Cars', 'MPG', 'step-3-mpg.pdf')
        self.create_dot_chart(self.__table, 2, 'Displacement of All Cars', 'Displacement', 'step-3-displacement.pdf')
        self.create_dot_chart(self.__table, 3, 'Horsepower of All Cars', 'Horsepower', 'step-3-horsepower.pdf')
        self.create_dot_chart(self.__table, 4, 'Weight of All Cars', 'Weight', 'step-3-weight.pdf')
        self.create_dot_chart(self.__table, 5, 'Acceleration of All Cars', 'Acceleration', 'step-3-acceleration.pdf')
        self.create_dot_chart(self.__table, 9, 'MSRP of All Cars', 'MSRP', 'step-3-msrp.pdf')

    def discretize_mpg_DoE(self, index):
        """Converts mpg into a categorical attribute using US Department of Energy ratings."""
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
                    
        #Resets the figure
        pyplot.figure()
    
        #Generates frequency diagram
        pyplot.bar(ratings, counts, align='center')
        pyplot.grid(True)
        pyplot.title('MPG Discretization Using Fuel Economy Ratings')
        pyplot.xticks(ratings)
    
        pyplot.savefig('step-4-approach1.pdf')
        pyplot.close()

    def discretize_mpg_bins(self, index):
        """Converts mpg into a categorical attribute using US Department of Energy ratings."""
        xs = self.get_column_as_floats(self.__table, index)
        counts = [0 for i in range(5)]
        bins = [i for i in range(5)]
        ratings = ['0-9', '10-18', '19-27', '28-36', '37-45']   
        #Creates frequency counts based on the ratings
        for i in range(len(xs)):
            if xs[i] <= 9:
                counts[0] += 1
            elif xs[i] <= 18:
                counts[1] += 1
            elif xs[i] <= 27:
                counts[2] += 1
            elif xs[i] <= 36:
                counts[3] += 1
            elif xs[i] <= 45:
                counts[4] += 1
            else:
                print 'tooo big man'
                exit(-1)

        #Resets the figure
        pyplot.figure()
    
        #Generates frequency diagram
        pyplot.bar(bins, counts, align='center')
        pyplot.grid(True)
        pyplot.title('MPG Discretization Using Equal Width Bins')
        pyplot.xticks(bins, ratings)
    
        pyplot.savefig('step-4-approach2.pdf')
        pyplot.close()
        
    def create_histogram(self, table, index, title, xlabel, outfile):
        xs = self.get_column_as_floats(table, index)
        
        pyplot.figure()
        pyplot.title(title)
        pyplot.xlabel(xlabel)
        pyplot.ylabel('Counts')
        pyplot.hist(xs)
        
        pyplot.savefig(outfile)
        pyplot.close()
        
    #mpg, cylinders, displacement, horsepower, weight, acceleration, model year, origin, and car name
    def create_all_histograms(self):
        self.create_histogram(self.__table, 0, 'Distribution of MPG Values', 'MPG', 'step-5-mpg.pdf')
        self.create_histogram(self.__table, 2, 'Distribution of Displacement Values', 'Displacement', 'step-5-displacement.pdf')
        self.create_histogram(self.__table, 3, 'Distribution of Horsepower Values', 'Horsepower', 'step-5-horsepower.pdf')
        self.create_histogram(self.__table, 4, 'Distribution of Weight Values', 'Weight', 'step-5-weight.pdf')
        self.create_histogram(self.__table, 5, 'Distribution of Acceleration Values', 'Acceleration', 'step-5-acceleration.pdf')
        self.create_histogram(self.__table, 9, 'Distribution of MSRP Values', 'MSRP', 'step-5-msrp.pdf')
     
    def create_scatter_plot(self, table, index, title, xlabel, outfile):
        ys = self.get_column_as_floats(table, 0)
        xs = self.get_column_as_floats(table, index)
        
        pyplot.figure()
        pyplot.title(title)
        pyplot.xlabel(xlabel)
        pyplot.ylabel('MPG')
        pyplot.grid(True)
        pyplot.plot(xs, ys, marker='.', linestyle='None')

        pyplot.savefig(outfile)
        pyplot.close()
        
    def create_all_scatter_plots(self):
        self.create_scatter_plot(self.__table, 2, 'Displacement vs. MPG', 'Displacement', 'step-6-displacement.pdf')
        self.create_scatter_plot(self.__table, 3, 'Horsepower vs. MPG', 'Horsepower', 'step-6-horsepower.pdf')
        self.create_scatter_plot(self.__table, 4, 'Weight vs. MPG', 'Weight', 'step-6-weight.pdf')
        self.create_scatter_plot(self.__table, 5, 'Acceleration vs. MPG', 'Acceleration', 'step-6-acceleration.pdf')
        self.create_scatter_plot(self.__table, 9, 'MSRP vs. MPG', 'MSRP', 'step-6-msrp.pdf')
        
    def calculate_mpg_by_year(self, table):
        table = sorted(table, key=operator.itemgetter(6))
        years, mpgValues = [], []
        
        for row in table:
            if row[6] not in years:
                years.append(row[6])
                mpgValues.append([float(row[0])])
            else:
                mpgValues[-1].append(float(row[0]))
        return years, mpgValues
            
    def create_boxplot(self):
        years, mpgValues = self.calculate_mpg_by_year(self.__table)
        
        pyplot.title('MPG by Model Year')
        pyplot.xlabel('Model Year')
        pyplot.ylabel('MPG')
        pyplot.grid(True)
        pyplot.xticks(years, years)
        pyplot.boxplot(mpgValues)
        pyplot.savefig('step-8-mpgbyyear.pdf')
    
def main():
    visualizationObject = DataVisualization()
    
    visualizationObject.create_all_freq_diagrams()
    visualizationObject.create_all_pie_charts()
    visualizationObject.create_all_dot_charts()
    visualizationObject.discretize_mpg_DoE(0)
    visualizationObject.discretize_mpg_bins(0)
    visualizationObject.create_all_histograms()
    visualizationObject.create_all_scatter_plots()
    visualizationObject.create_boxplot()

main()
    






