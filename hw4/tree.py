       
class Tree:
    def __init__(self, nodeType, nodeValue):
        self.nodeType = nodeType
        self.nodeValue = nodeValue
        self.subtrees = []
        
    
    def append(self, subtree):
        self.subtrees.append(subtree)
        	
        
    def __str__(self, level=0):
    	display = "|" + "--" * level + "|" + str(self.nodeType) + " " + str(self.nodeValue) + "\n"
    	for subtree in self.subtrees:
    		display += subtree.__str__(level + 1)
    	return display


def main():
    t = Tree('attribute', 0)
    t.append(Tree('value', 1))
    t.append(Tree('value', 2))
    t.append(Tree('value', 3))
    t.subtrees[0].append(Tree('attribute', 1))
    t.subtrees[0].subtrees[0].append(Tree('value', '1'))
    t.subtrees[0].subtrees[0].subtrees[0].append(Tree('yes', [1, 1]))
    print t


if __name__ == '__main__':
    main()

