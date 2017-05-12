import sys
import json
import os

if __name__ == "__main__":
    AGGREGATE_RESULTS_FOLDER = 'aggregate_results_groups'
    AGGREGATE_RESULTS_SUFFIX = 'aggregate_results.json'

    if len(sys.argv) == 2:

        if not os.path.exists(AGGREGATE_RESULTS_FOLDER):
            os.makedirs(AGGREGATE_RESULTS_FOLDER)

        with open(sys.argv[1], 'r') as file:
            claim_urls = json.load(file)
        group_id = 1
        tmp_data = []
        count = 0
        for idx, claim_url in enumerate(claim_urls):
            if count % 15 == 0 and count != 0:
                print len(tmp_data)
                with open(AGGREGATE_RESULTS_FOLDER+"/"+str(group_id)+"_"+AGGREGATE_RESULTS_SUFFIX, 'w') as file:
                    json.dump(tmp_data, file, indent=2)
                group_id += 1
                tmp_data = []
            tmp_data.append(claim_url)
            count += 1

        # Store the rest
        with open(AGGREGATE_RESULTS_FOLDER + "/" + str(group_id) + "_" + AGGREGATE_RESULTS_SUFFIX, 'w') as file:
            print len(tmp_data)
            json.dump(tmp_data, file, indent=2)

        print len(claim_urls)
    else:
        print "please specify where the aggregate_results is "