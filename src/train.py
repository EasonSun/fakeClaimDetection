
#read in data and set fold 
if __name__ == "__main__":
	dataPath = sys.argv[1]
    claimList = io.readClaim(dataPath)
    # train lgClassifier
    lgClassifier(claimList)
    lgClassifier.train()
    # same for the other 2, later