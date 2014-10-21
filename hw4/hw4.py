
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
	
    def k_cross_fold_partition(self, table, k, classIndex, curBin):
        """Partition a dataset into training and test by splitting into K folds (bins)."""
        # randomize 
        randomized = table # table passed is already a copy
        n = len(table)
       
        for i in range(n):
            # pick an index to swap
            j = random.randint(0, n-1) # random int [0,n-1] inclusive
            randomized[i], randomized[j] = randomized[j], randomized[i]

        # get classes
        classNames = []
        for row in randomized:
            if row[classIndex] not in classNames:
                classNames.append(row[classIndex])
        
        # partition dataset - each subset contains rows with a unique class
        dataPartition = []
        for i in range(len(classNames)):
            dataPartition.append(\
                self.partition_classes(classIndex, classNames[i], randomized))

        # distribute paritions roughly equally
        kPartitions = [[] for _ in range(k)]
        for partition in dataPartition:
            for i in range(len(partition)):
                kPartitions[i%k].append(partition[i])
        
        # return training and test set
        testSet = kPartitions[curBin]
        trainingSet = []
        for i in range(k):
            if i != curBin:
                trainingSet += kPartitions[i]
        return trainingSet, testSet
	
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
		
	def select_attribute(self, instances, attIndices, classIndex):
		'''Returns attribute index to partition on.'''
        # random selection for now
		pass
		
	
	def resolve_clash(self, otherParams):
		'''.'''
		pass
	
    def tdidt(self, instances, attIndices, classIndex):
		'''Returns tree object.
           Uses Top Down Induction of Decision Trees recursive algorithm.
		   Algorithm:
			- At each step, pick an attribute (“attribute selection”)
			- Partition data by attribute values ... pairwise disjoint partitions
			- Repeat until (base cases):
				 1. Partition has only class labels that are the same ... no clashes
				 2. No more attributes to partition ... there may be clashes
				 3. No more instances to partition ... backtrack, create single leaf node
        '''
        # TODO: understand wtf is going on ---v       
		# Repeat until base case(s)
		# Only class labels that are the same
        if self.in_same_class(instances, self.classIndex) == True:
            return 
		# No more instances to partition
        if len(instances) == 0:
            return
		# No more attributes to partition
        if numAtt == 0:
            return       

        # At each step select an attribute
        curAtt = self.select_attribute(instances, attIndices)
		
		# Partition data by attribute values
		attVals = self.partition_instances(instances, attIndex)

    def dt_titanic(table):
        """Creates a decision tree for titanic.txt and classifies instances
           according to the generated tree for each k in the k-fold cross validation.
           Creates confusion matrices for the results and compares to HW3 classifiers."""
        k = 10
        for curBin in range(len(k)):
            train, test =  self.k_cross_fold_partition(table, k, self.classIndex, curBin):

            # build tree with training set
            dt = self.tdidt(train, attIndices)

            # classify test set using tree
            for instance in test:
                classified.append(self.tdidt_classify(dt, instance, attIndices))
                actual.append(instance[classIndex])
            
            # generate matrix

def main():
    """Hello."""
    tableCopy = copy.deepcopy(self.table)
    t = DecisionTreeClassifier(tableCopy)		
	t.dt_titanic(tableCopy)



if __name__ == "__main__":
    main()

