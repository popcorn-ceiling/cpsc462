       
class Tree:
    def __init__(self, nodeLabel, nodeValue):
        self.nodeLabel = nodeLabel
        self.nodeValue = nodeValue
        self.subtrees = []
        
    
    def append(self, nodeLabel, nodeValue):
        self.subtrees.append(Tree(nodeLabel, nodeValue))
        	
        
    def __str__(self, level=0):
    	display = "|" + "--" * level + "|" + str(self.nodeLabel) + " " + str(self.nodeValue) + "\n"
    	for subtree in self.subtrees:
    		display += subtree.__str__(level + 1)
    	return display


def main():
    t = Tree('attribute', 0)
    t.append('value', 1)
    t.append('value', 2)
    t.append('value', 3)
    t.append('value', 4)
    t.subtrees[0].append('attribute', 1)
    t.subtrees[0].subtrees[0].append('value', '1')
    t.subtrees[0].subtrees[0].subtrees[0].append('yes', [1, 1])
    t.subtrees[1].append('attribute', 2)
    t.subtrees[3].append('attribute', 5)
    t.subtrees[3].subtrees[0].append('value', 1)
    print t


if __name__ == '__main__':
    main()

