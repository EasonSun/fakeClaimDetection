from os import listdir
from os.path import isfile, join
import numpy as np
import json
import random
import math


class Network:
    snope_folder = ''

    # claim, topic, source, and stance feature dict
    claimId = {}
    claimMaxId = -1
    sourceId = {}
    sourceMaxId = -1
    stanceFeatureId = {}
    stanceFeatureMaxId = -1
    topicId = {}
    topicMaxId = -1

    claimLabelId = {}
    sourceLabelId = {}
    stanceFeatureLabelId = {}

    # CS, CT, CF of type {"claim id ": [sourceid1, sourceid2], ... }
    CSmatrix = {}
    CTmatrix = {}
    CFmatrix = {}

    def __init__(self, snopesfolder=None):
        self.snope_folder = snopesfolder if snopesfolder else "Snopes"

        self.dataToId()
        with open('claim.json', 'w') as file:
            json.dump(self.claimId, file, indent=2)
        with open('source.json', 'w') as file:
            json.dump(self.sourceId, file, indent=2)
        with open('CS.json', 'w') as file:
            json.dump(self.CSmatrix, file, indent=2)

        #print(self.claimMaxId + 1)
        #print(self.sourceMaxId + 1)

    def dataToId(self):
        claimfiles = [f for f in listdir(self.snope_folder) if isfile(join(self.snope_folder, f))]
        #print(len(claimfiles))

        claims = []  # --------------------
        for claim_file in claimfiles:
            with open(self.snope_folder + '/' + claim_file) as file:
                claim_data = json.load(file)
                claim_str = claim_data["Claim"]
                source_str = claim_data["Referred Links"]
                claim_ID = claim_data["Claim_ID"]  # ---------
                claims.append({"claim": claim_str, "claimID": claim_ID})  # -------
                if claim_str not in self.claimId:
                    self.claimId[claim_str] = self.claimMaxId + 1
                    self.claimMaxId += 1

                sources = [s for s in source_str.strip().split(';') if s.strip() != ""]

                for source in sources:
                    if source not in self.sourceId:
                        self.sourceId[source] = self.sourceMaxId + 1
                        self.sourceMaxId += 1

                # Add relation between claim and source
                related_source = set()
                for source in sources:
                    related_source.add(self.sourceId[source])
                self.CSmatrix[self.claimId[claim_str]] = list(related_source)

        with open("all_claims.json", 'w') as file:  # ----------------
            json.dump(claims, file, indent=2)


    '''
    #####   claim related
    '''
    # Return -1 if claim_str does not exist in dict
    def getClaimId(self, claim_str):
        if claim_str in self.claimId:
            return self.claimId[claim_str]
        else:
            return -1

    # This will add the claim and return its ID
    def addClaim(self, claim_str):
        if claim_str in self.claimId:
            return self.claimId[claim_str]
        else:
            self.claimId[claim_str] = self.claimMaxId + 1
            self.claimMaxId += 1

    def getClaimSize(self):
        return self.claimMaxId+1

    '''
    #####   source related
    '''
    # Return -1 if source does not exist in dict
    def getSourceId(self, source_str):
        if source_str in self.sourceId:
            return self.sourceId[source_str]
        else:
            return -1

    # This will add the source and return its ID
    def addSource(self, source_str):
        if source_str in self.sourceId:
            return self.sourceId[source_str]
        else:
            self.sourceId[source_str] = self.sourceMaxId + 1
            self.sourceMaxId += 1

    def getSourceSize(self):
        return self.sourceMaxId+1
    '''
    #####   topic related
    '''
    # Return -1 if source does not exist in dict
    def getTopicId(self, topic):
        if topic in self.topicId:
            return self.topicId[topic]
        else:
            return -1

    # This will add the source and return its ID
    def addTopic(self, topic):
        if topic in self.topicId:
            return self.topicId[topic]
        else:
            self.topicId[topic] = self.topicId + 1
            self.topicMaxId += 1

    def getTopicSize(self):
        return self.topicMaxId+1
    '''
    #####  stance feature related
    '''
    # Return -1 if source does not exist in dict
    def getStanceFeatureId(self, stancef):
        if stancef in self.stanceFeatureId:
            return self.stanceFeatureId[stancef]
        else:
            return -1

    # This will add the source and return its ID
    def addStanceFeature(self, stancef):
        if stancef in self.stanceFeatureId:
            return self.stanceFeatureId[stancef]
        else:
            self.stanceFeatureId[stancef] = self.stanceFeatureId + 1
            self.stanceFeatureMaxId += 1

    def getStanceFeatureSize(self):
        return self.stanceFeatureMaxId+1

    '''
    ### label related, 0 means true, 1 means false, -1 means no label
    '''
    def addClaimLabel(self, claim_id, label):
        self.claimLabelId[claim_id] = label
    
    def getClaimLabel(self, claim_id, label):
        return self.claimLabelId.get(claim_id, -1)

    def addSourceLabel(self, source_id, label):
        self.sourceLabelId[source_id] = label
    
    def getSourceLabel(self, source_id, label):
        return self.sourceLabelId.get(source_id, -1)

    def addStanceFeatureLabel(self, stance_id, label):
        self.stanceFeatureLabelId[stance_id] = label
    
    def getStanceFeatureLabel(self, stance_id, label):
        return self.stanceFeatureLabelId.get(stance_id, -1)

    '''
    ### Get relation matrices 
    '''
    # Return relation matrix in np array
    def getClaimSourceMatrix(self):
        claim_size = self.claimMaxId+1
        source_size = self.sourceMaxId+1
        return self.__getMatrix(self.CSmatrix, claim_size, source_size, "claim_source")

    def getClaimTopicMatrix(self):
        claim_size = self.claimMaxId + 1
        topic_size = self.topicMaxId + 1
        return self.__getMatrix(self.CTmatrix, claim_size, topic_size, "claim_topic")

    def getClaimStanceFeatureMatrix(self):
        claim_size = self.claimMaxId + 1
        feature_size = self.stanceFeatureMaxId + 1
        return self.__getMatrix(self.CFmatrix, claim_size, feature_size, "claim_feature")

    def __getMatrix(self, relation_matrix, dim1, dim2, filename=None):
        matrix = np.zeros((dim1, dim2))
        for i in range(dim1):
            for j in range(dim2):
                matrix[i, j] = 1 if j in relation_matrix[i] else 0
        if filename:
            np.save(filename, matrix)
        return matrix





if __name__ == "__main__":
    network = Network("Snopes")
    csmatrix = network.getClaimSourceMatrix()
    original_csmatrix = network.CSmatrix
    # testing
    for i in range(1000):
        claim_id = random.randrange(0, network.getClaimSize())
        source_id = random.randrange(0, network.getSourceSize())
        both_true =  (csmatrix[claim_id, source_id] == 1) and source_id in original_csmatrix[claim_id]
        both_false = (csmatrix[claim_id, source_id] == 0) and source_id not in original_csmatrix[claim_id]
        if both_true or both_false:
            pass
        else:
            print("Matrix is WRONG!")


    