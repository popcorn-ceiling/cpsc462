
import matplotlib.pyplot as pyplot
import numpy 
import csv
import operator
import numpy

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
    
    def average(self, vals):
        if len(vals) != 0:
            return round(float(sum(vals)/len(vals)), 2)
        else:
            return 0

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
        """Creates a dot chart for a given continuous attribute."""
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
        """Creates dot charts of all continuous attributes of auto-data.txt dataset."""
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
                        
        #Generates frequency diagram
        pyplot.figure()
        pyplot.bar(ratings, counts, align='center')
        pyplot.grid(True)
        pyplot.title('MPG Discretization Using Fuel Economy Ratings')
        pyplot.xticks(ratings)
    
        pyplot.savefig('step-4-approach1.pdf')
        pyplot.close()

    def discretize_mpg_bins(self, index):
        """Converts mpg into a categorical attribute using equal width binds."""
        xs = self.get_column_as_floats(self.__table, index)
        counts = [0 for i in range(5)]
        bins = [i for i in range(5)]
        ratings = ['0-9', '10-18', '19-27', '28-36', '37-45']   
        
        #Creates frequency counts based on the range of the bins
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
    
        #Generates frequency diagram
        pyplot.figure()
        pyplot.bar(bins, counts, align='center')
        pyplot.grid(True)
        pyplot.title('MPG Discretization Using Equal Width Bins')
        pyplot.xticks(bins, ratings)
    
        pyplot.savefig('step-4-approach2.pdf')
        pyplot.close()
        
    def create_histogram(self, table, index, title, xlabel, outfile):
        """Creates a histogram for a continuous attribute."""
        xs = self.get_column_as_floats(table, index)
        
        pyplot.figure()
        pyplot.title(title)
        pyplot.xlabel(xlabel)
        pyplot.ylabel('Counts')
        pyplot.hist(xs)
        
        pyplot.savefig(outfile)
        pyplot.close()
        
    def create_all_histograms(self):
        """Creates histograms of all continuous attributes of auto-data.txt dataset."""
        self.create_histogram(self.__table, 0, 'Distribution of MPG Values', 'MPG', 'step-5-mpg.pdf')
        self.create_histogram(self.__table, 2, 'Distribution of Displacement Values', 'Displacement', 'step-5-displacement.pdf')
        self.create_histogram(self.__table, 3, 'Distribution of Horsepower Values', 'Horsepower', 'step-5-horsepower.pdf')
        self.create_histogram(self.__table, 4, 'Distribution of Weight Values', 'Weight', 'step-5-weight.pdf')
        self.create_histogram(self.__table, 5, 'Distribution of Acceleration Values', 'Acceleration', 'step-5-acceleration.pdf')
        self.create_histogram(self.__table, 9, 'Distribution of MSRP Values', 'MSRP', 'step-5-msrp.pdf')
     
    def create_scatter_plot(self, table, index, title, xlabel, outfile):
        """Creates a scatter plot to compare an attribute to mpg."""
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
        """Creates scatter plots that compare displacement, horsepower, weight, acceleration, \
            and msrp to mpg."""
        self.create_scatter_plot(self.__table, 2, 'Displacement vs. MPG', 'Displacement', 'step-6-displacement.pdf')
        self.create_scatter_plot(self.__table, 3, 'Horsepower vs. MPG', 'Horsepower', 'step-6-horsepower.pdf')
        self.create_scatter_plot(self.__table, 4, 'Weight vs. MPG', 'Weight', 'step-6-weight.pdf')
        self.create_scatter_plot(self.__table, 5, 'Acceleration vs. MPG', 'Acceleration', 'step-6-acceleration.pdf')
        self.create_scatter_plot(self.__table, 9, 'MSRP vs. MPG', 'MSRP', 'step-6-msrp.pdf')

    def calculate_least_squares_lr(self, xs, ys):
        xAvg = self.average(xs)
        yAvg = self.average(ys)

        # calculate m, slope of line
        mTop = 0
        mBot = 0
        for i in range(len(xs)):
            mTop += ((xs[i] - xAvg)*(ys[i] - yAvg)) 
            mBot += (xs[i] - xAvg)**2
        m = float(mTop / mBot)

        # calculate b, y intercept of line
        b = yAvg - (m * xAvg)

        return m, b
    
    def calculate_covariance(self, xs, ys):
        xAvg = self.average(xs)
        yAvg = self.average(ys)
           
        cov_sum = 0
        for i in range(len(xs)):
            cov_sum += (xs[i] - xAvg)*(ys[i] - yAvg)

        return float(cov_sum / len(xs))

    def calculate_correlation_coefficient(self, xs, ys, cov):
        stdx = numpy.std(xs)
        stdy = numpy.std(ys)
    
        return float(cov/(stdx*stdy))

    def create_linear_regression_plot(self, table, index0, index1, xlabel, ylabel, outfile):
        xs = self.get_column_as_floats(table, index0)
        ys = self.get_column_as_floats(table, index1)
        m, b = self.calculate_least_squares_lr(xs, ys)
        cov = round(self.calculate_covariance(xs, ys), 2)
        corr_coeff = round(self.calculate_correlation_coefficient(xs, ys, cov), 2)

        title = xlabel + ' vs. ' + ylabel
        textbox = ' corr: ' + str(corr_coeff) + ' cov: ' + str(cov)
        txtX = max(xs) - 500
        txtY = max(ys) - 50

        pyplot.figure()
        pyplot.title(title)
        pyplot.xlabel(xlabel)
        pyplot.ylabel(ylabel)
        pyplot.grid(True)

        # scatter plot
        pyplot.plot(xs, ys, marker='.', linestyle='None')

        # linear regression 
        xSort = sorted(xs)
        x_r = numpy.arange(xSort[0], xSort[-1], 1)
        pyplot.plot(x_r, (x_r * m) + b, "r")

        # textbox with correlation coefficient and covariance
        box = dict(facecolor='none', color='r')
        pyplot.annotate(textbox, bbox=box, color='r', xy=(0.35, 0.95), xycoords='axes fraction') 

        pyplot.savefig(outfile)
        pyplot.close()
    
    def create_all_linear_regression_plots(self):
        
        self.create_linear_regression_plot(self.__table, 2, 0, 'Displacement', 'MPG', 'step-7-displacement.pdf')
        self.create_linear_regression_plot(self.__table, 3, 0, 'Horsepower', 'MPG', 'step-7-horsepower.pdf')
        self.create_linear_regression_plot(self.__table, 4, 0, 'Weight', 'MPG', 'step-7-weight.pdf')
        self.create_linear_regression_plot(self.__table, 5, 0, 'Acceleration', 'MPG', 'step-7-acceleration.pdf')
        self.create_linear_regression_plot(self.__table, 9, 0, 'MSRP', 'MPG', 'step-7-msrp.pdf')
        self.create_linear_regression_plot(self.__table, 4, 2, 'Weight', 'Displacement', 'step-7-wght-dsp.pdf')
        
    def calculate_mpg_by_year(self, table):
        """Calculates the mpg values based on year values \
            Helper function for create_boxplot()."""
        
        #Sorts the table by year
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
        """Create a box plot describing MPG by year."""
        years, mpgValues = self.calculate_mpg_by_year(self.__table)
        
        pyplot.title('MPG by Model Year')
        pyplot.xlabel('Model Year')
        pyplot.ylabel('MPG')
        pyplot.grid(True)
        pyplot.xticks(years, years)
        pyplot.boxplot(mpgValues)
        pyplot.savefig('step-8-mpgbyyear.pdf')
        
    def group_by(self, table, att_index):
        """Partitions the rows of the given table by the attribute."""
        #Creates unique, sorted list of grouping values
        grouping_values = []
        for row in table:
            value = row[att_index]
            if value not in grouping_values:
                grouping_values.append(value)
        grouping_values.sort()
        
        #Creates list of n empty partitions
        results = [[] for _ in range(len(grouping_values))]

        #Adds rows to each partition
        for row in table:
            results[grouping_values.index(row[att_index])].append(row[:])
        
        return results, grouping_values
        
    def create_multiple_freq_diagrams(self):
        """Creates a frequency diagram of the number of cars from each country of origin \
            separated out by model year."""
        
        #ISSUE: the separate bar graphs do not correctly line up over the corresponding years
        
        #Gets the grouped tables to be plotted and the x labels
        grouped_table, grouping_values = self.group_by(self.__table, 6)
        
        #Gets the lists of origin values for each table grouping
        xs_lists = []
        for group in grouped_table:
            xs = self.get_column_as_floats(group, 7)
            xs_lists.append(xs)
        
        #Gets the values and counts for each list of origin values
        count_list = []
        for xs in xs_lists:
            values, counts = self.calculate_frequencies(xs)
            count_list.append(counts)
            
        #Groups the counts by origin number
        origin_1_counts = []
        origin_2_counts = []
        origin_3_counts = [] 
        for item in count_list:
              origin_1_counts.append(item[0])
              origin_2_counts.append(item[1])
              origin_3_counts.append(item[2])

        
        fig, ax = pyplot.subplots()
        bar_width = 0.3
        x_locations = numpy.arange(len(count_list))
        
        r1 = ax.bar(x_locations, origin_1_counts, bar_width, color='b', align='center')
        r2 = ax.bar(x_locations + bar_width, origin_2_counts, bar_width, color='g', align='center')
        r3 = ax.bar(x_locations + 2*bar_width, origin_3_counts, bar_width, color='r', align='center')
        
        ax.set_xticklabels(grouping_values)
        ax.legend((r1[0], r2[0], r3[0]), ('US', 'Europe', 'Japan'))
        
        #pyplot.show()
        
    
def main():
    visualizationObject = DataVisualization()
    
   # visualizationObject.create_all_freq_diagrams()
   # visualizationObject.create_all_pie_charts()
   # visualizationObject.create_all_dot_charts()
   # visualizationObject.discretize_mpg_DoE(0)
   # visualizationObject.discretize_mpg_bins(0)
   # visualizationObject.create_all_histograms()
   # visualizationObject.create_all_scatter_plots()
   # visualizationObject.create_boxplot()
    visualizationObject.create_all_linear_regression_plots()
   # visualizationObject.create_multiple_freq_diagrams()
    

main()
    
