experimentPath = sys.argv[1]
claimPath = experimentPath + 'claims.txt'
credPath = experimentPath + 'cred.npy'


def main():
	claims = []
    with open(claimPath) as f:
        claims = f.readlines()
    claims = [x.strip() for x in claims] 	
    claimX = []

    for claim in claims:
    	articles = articleCrawl(claim)
    	claimX.append(extractFeatures(claim, articles))

    


from sklearn.externals import joblib
estimatorPath = experimentPath + 'rf.pkl'
rf = joblib.load(estimatorPath)

