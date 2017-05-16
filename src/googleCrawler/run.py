import json
import os
import subprocess
import sys
import io

if __name__ =="__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print dir_path
    os.chdir(dir_path)
    AGGREGATE_RESULTS_FOLDER = 'aggregate_results_groups'
    AGGREGATE_RESULTS_SUFFIX = 'aggregate_results.json'
    aggregate_results_file = 'aggregate_results.json'
    final_results_folder = 'final_results_ver2'
    # snopes = 'Snopes'

    lower_group_id = 1
    higher_group_id = 1

    if not os.path.exists(final_results_folder):
        os.makedirs(final_results_folder)

    if len(sys.argv) == 2 or len(sys.argv) == 3:
        if len(sys.argv) == 2:
            lower_group_id = int(sys.argv[1])
            higher_group_id = lower_group_id
        else:
            lower_group_id = min(int(sys.argv[1]), int(sys.argv[2]))
            higher_group_id = max(int(sys.argv[1]), int(sys.argv[2]))

        for id in range(lower_group_id, higher_group_id+1):
            results = []
            with io.open(AGGREGATE_RESULTS_FOLDER+"/"+str(id)+"_"+AGGREGATE_RESULTS_SUFFIX, 'r') as file:
                results = json.load(file)
            for result in results:
                subprocess.call(["scrapy", "runspider", "-a", "aggregate_results_file=" + aggregate_results_file,
                                 "-a", "claim_file=" + result['claim_file'],"--nolog", "-o",
                                 final_results_folder + "/" + result['claim_file'], "articles_spider_ver2.py"], shell=False)
                print subprocess.list2cmdline(
                    ["scrapy", "runspider", "-a", "aggregate_results_file=" + aggregate_results_file,
                     "-a", "claim_file=" + result['claim_file'], "-o", final_results_folder + "/" + result['claim_file'],
                     "articles_spider_ver2.py"])


    else:
        print "please pass in group id"
        exit()
