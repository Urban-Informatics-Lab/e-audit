class BuildingFeatures: 
    def __init__(self, alg):
        # alg is a string that can be 'Euclidean' or 'KNN' or 'Decision Tree'
        self.alg = alg
    
    # process_alg takes the algorithm input and calls the appropriate method
    def process_alg(self):
        if self.alg == 'Euclidean':
            self.Euclidean()
        elif self.alg == 'KNN':
            self.KNN_classifiers()
        elif self.alg == 'Decision Tree':
            self.DT_classifiers()
        else: 
            print("Invalid Algorithm Input. Please provide 'Euclidean', 'KNN', or 'Decision Tree.'")
    
    def Euclidean(self):
        print("Calculating Euclidean Distance")
    
    def KNN_classifiers(self): 
        print("Calculating KNN")

    def DT_classifiers(self): 
        print("Calculating the Decision Tree(s)")

#create an instance for each algorithm type for testing 
bf = BuildingFeatures('Euclidean')
bf.process_alg()

bf1 = BuildingFeatures('KNN')
bf1.process_alg()

bf2 = BuildingFeatures('Decision Tree')
bf2.process_alg()

bf3 = BuildingFeatures('ML')
bf3.process_alg()