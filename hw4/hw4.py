
from tree import Tree

class DecisionTreeClassifier:

	def __init__(self, fileName, classIndex):
		self.table = read_csv(fileName)
		self.classIndex = classIndex
		self.decisionTree = None

	def read_csv(self, fileName):
        """Reads in a csv file and returns a table as a list of lists (rows)."""
        theFile = open(fileName)
        theReader = csv.reader(theFile, dialect='excel')
        table = []
        for row in theReader:
            if len(row) > 0:
                table.append(row)
        return table
	
	
	
	def tdidt(self, instances, attIndexes, classIndex):
		'''Uses Top Down Induction of Decision Trees recursive algorithm.
		   Algorithm:
			- At each step, pick an attribute (“attribute selection”)
			- Partition data by attribute values ... pairwise disjoint partitions
			- Repeat until (base cases):
				 1. Partition has only class labels that are the same ... no clashes
				 2. No more attributes to partition ... there may be clashes
				 3. No more instances to partition ... backtrack, create single leaf node
        '''
        
        #At each step select an attribute
            self.select_attribute(instances, attIndexes)
		
		# Partition data by attribute values
		    self.partition_instances(instances, attIndex)
		    
		# Repeat until base case(s)
		    #Only class labels that are the same
		        self.in_same_class(instance)
		    #No more attributes to partition
		        ......
		    #No more instances to partition
		        ......
		
		
	def in_same_class(self, instances):
		'''Returns true if all instances have same label.'''
		# Get first label
		testLabel = instances[0][self.classIndex]
		
		# Test whether all instances have the same label
		for instance in instances:
			if instance[self.classIndex] != testLabel:
				return False
				
		return True
		
		
	def partition_stats(self, instances, classIndex):
		'''List of stats: [[label1, occ1, tot1], [label2, occ2, tot2], ...].'''
		pass
		
		
	def partition_instances(instances, attIndex):
		'''Partition list: {attval1:part1, attval2:part2}.'''
		
		partition1 = {}
		for i in range(attIndex + 1):
		    
		    
		
		
		
		
	def select_attribute(self, instances, attIndexes, classIndex):
		'''Returns attribute index to partition on.'''
		pass
		
	
	def resolve_clash(self, otherParams):
		'''.'''
		pass
		
	