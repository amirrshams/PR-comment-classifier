#!/usr/bin/env python3
# Categories comments via a set of heuristics in a text file
#
# The goal of this code is to perform a heuristics search of all files and do the following:
# 1. Categorize comments by keywords or base words
# 2. Add a new column - comment-group on output
# 3. Write out parsed comment line
# 4. Reach conclusions -- non-code tasks

import os.path
import time
import datetime
from random import Random
import csv
import sys
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

#Define constants
KEYWORD_DELIMITER = ":"
COMMENT_DELIMITER = ","
EMPTY_COMMENT = "[]"
MAX_DOT_PER_LINE = 100
MAX_RECORD_PER_DOT = 80
MERGED_STATUS = "merged"

#Set field size for csv to be large enough
csv.field_size_limit(sys.maxsize)

#Define the stemmer
wl = WordNetLemmatizer()
ps = PorterStemmer()
rnd = Random()


#Represents a single pull request instance
class PullRequestMetric:
    pr_id = ''
    comments = ''
    status = ''
    author_id = ''
    comment_group = ''
    manual_analysis = ''
    pr_url = ''

    def __init__(self, pr_id, status, author_id, pr_url, comments, manual_analysis, comment_group, heuristic_correctness):
        self.pr_id = pr_id
        self.status = status
        self.author = author_id
        self.pr_url = pr_url
        self.comments = comments
        self.manual_analysis = manual_analysis
        self.comment_group = comment_group
        self.heuristic_correctness = heuristic_correctness
        

    def get_comments(self):
        print('retrieving comments')
        return  self.comments

    def get_csv(self):
        out = [self.pr_id, self.status, self.author, self.pr_url, self.comments, self.manual_analysis, self.comment_group, self.heuristic_correctness]
        return out


#Keywords dictionary
class Keywords:
    words = []
    category = ''
    def __init__(self, category, words):
        self.words = words
        self.category = category

    def get_words(self):
        return self.words

    def get_category(self):
        return self.category

# Opens the keyword input file and load it into memory
def open_mapping_file(kywrd_file):

    #a dictionary of keywords and their mapping
    keywords = {}
    if kywrd_file  != '' and os.path.isfile(kywrd_file):
        with open(kywrd_file, newline='') as f:
            kw_file = f.readlines()
            for row in kw_file:
                #print(row)
                v = row.split(KEYWORD_DELIMITER)

                #Ignore comments and blank lines
                if len (v) == 2:
                    #Parse the well-defined columns - col 0 - keywords, col 1 - grouping
                    #This code must be this way as the line format is x1, x2,,,,xn :k1

                    key = v[1].strip()
                    value =  v[0].strip()
                    ky = Keywords(key, value)
                    # keywords.add(ky)
                    keywords[key] = ky
    return keywords

#Checks if a text or string is blank
def is_blank (a_text):
    return not (a_text and a_text.strip())

def is_not_blank(a_text):
    return not is_blank(a_text)

#A comment is considered empty if it contains '[]' only or is blank
def is_empty (txt):
    if is_blank(txt) or EMPTY_COMMENT == txt.strip():
        return True
    # else:
    #     if EMPTY_COMMENT == txt.strip():
    #         return  True
    return  False

#Check for word within comments, if none, then find the stem and return comment group
#Search where keyword belongs to more than 1 comment_group. This is fixed as part of
#Issue#10 -- Improve group label logic
#Algorithm: Find how many times a word within a comment label group is found in a comment
#   The label with the highest word count, wins.
def search_in_comment(keywords, comments_count, comments):

    #word_group = 'none'
    no_comment_group = 'No reason'

    comment_entry = comments.lower()

    if comments_count == 0 or is_empty(comment_entry):
        return no_comment_group

    keywords_keys = keywords.keys()

    #Store how many words have been found for a particular comment group mapping
    key_word_count = {}

    for keys in keywords_keys:
        kw = keywords[keys]
        words = kw.words.split(",")  #An array of words
        for w in words:
            w = w.strip()
            s = comment_entry.find(w)
            if s != -1 and s is not None:
                key_word_count[keys] = update_group_count(key_word_count, keys)
            else:
                #base_word = ps.stem(w)
                s = comment_entry.find(w)
                if s != -1 and s is not None:
                    key_word_count[keys] = update_group_count(key_word_count, keys)
                else:
                    base_word2 = wl.lemmatize(w,"v")
                    s = comment_entry.find(base_word2)
                    if s != -1 and s is not None:
                        key_word_count[keys] = update_group_count(key_word_count, keys)
                    #endif
                #endif
            #end-if
        #end-for
    #end-for
    word_group = assign_comment_group(key_word_count)

    #Clear out the dictionary for next run
    key_word_count.clear()

    return word_group

#Increase counter for words of a particular comment_group
def update_group_count(key_word_count, keys):
    c = 0
    try:
        if not keys in key_word_count:
            c = 1
        else:
            c = key_word_count[keys] + 1
        #end-if
    except KeyError:
        c = 0
    return c

#Go through the working memory for updated counts and select the max size or only.
#If they are all the same, then randomly pick one since any random value is valid
def assign_comment_group(key_word_mapping):
    comment_grp = "Successful"
    mx_v = 1
    mx_k = ''

    word_counts = len(key_word_mapping)
    if word_counts == 1:
        mx_k = next(iter(key_word_mapping.keys()))
        comment_grp = mx_k
        return comment_grp

    same_frequency_value = True
    for k in key_word_mapping:
        wc = key_word_mapping[k]
        if  wc > mx_v:
            mx_v = wc
            mx_k = k
            same_frequency_value = False

    #Pick the first one
    if same_frequency_value:
        keys_list = list(key_word_mapping.keys())
        word_counts = len(key_word_mapping)
        if word_counts > 1:
            index = rnd.randrange(0, word_counts)
            #mx_k = keys_list[-1]
            #mx_k = keys_list[0]
            mx_k = keys_list[index]


    if mx_k != '':
        comment_grp = mx_k
    return comment_grp


# The PR or comments file is processed one line at a time as it could be very large
# Steps::
# 3. Open comment file (process 1 line at a time)
# 4. For each line of comment,
#   4.1 Check if keyword exist, if none, check for base word (lemmatization)
#   4.2 If False, then skip and assign to None
#   4.3 If True, assign key word to target
#   4.4 Write output values in another file
def process_pull_requests(keywords, pr_comment_file):

    interim_file = generate_output_filename()
    header = ['pr_id', 'status', 'author_id', 'pr_url', 'comments', 'manual_category', 'comment_group', 'heuristic_correctness']

    #Open the output file
    if pr_comment_file != '' and os.path.isfile(pr_comment_file):
        with open(interim_file,'w', newline ="") as out_file:
            csv_output = csv.writer(out_file, dialect ="excel")
            csv_output.writerow(header)

            #Open the raw data input
            with open(pr_comment_file,"r") as csv_file:
                read_csv_line = csv.DictReader(csv_file)

                records_per_dot = 0
                dots_per_line = 0
                #Extract contents of each CSV line
                for row in read_csv_line:
                    # repo_id = row['repo_id']
                    status = row['status']
                    pr_id = row['pr_id']
                    author_id = row['author_id']
                    comments_count = row['comments_counts']
                    comments = row['comments']
                    pr_url = row['pr_url']
                    manual_analysis = row['manual_analysis']

                    
                    #Assign PR to a comment_group based on content of comments
                    if status == MERGED_STATUS:
                        comment_group = "Merged"
                    else:
                        comment_group = search_in_comment(keywords, comments_count, comments)

                    if manual_analysis == comment_group:
                        heuristic_correctness = 1
                    else:
                        heuristic_correctness = 0
                    if  is_not_blank(pr_id) and is_not_blank(status):
                        pr = PullRequestMetric(pr_id, status, author_id , pr_url, comments, manual_analysis, comment_group, heuristic_correctness)

                        #Added for manual verification of tests results
                        csv_output.writerow(pr.get_csv())

                        #Manage how many records are processed before a dot is printed
                        records_per_dot += 1
                        if  records_per_dot > MAX_RECORD_PER_DOT:
                            dots_per_line += 1
                            print(".", end="")
                            records_per_dot = 0

                        #How many dots are printed per line Advance to the next line
                        if dots_per_line > MAX_DOT_PER_LINE:
                            print("", end="\n")
                            dots_per_line = 0

                    comment_group = ''

                    


    return


#Generate an output filename based on time. This ensures that the report can be run several times over
#and each response kept
def generate_output_filename():
    timestamp = time.time()
    date_time = datetime.datetime.fromtimestamp(timestamp)
    part_file_name = date_time.strftime("%Y-%b-%d-%I-%M-%m")
    interim_file = "./heuristics_out_" + str(part_file_name) + ".csv"

    return interim_file

# Entry point into this code
if __name__ == '__main__':
    #Check input parameters
    if len(sys.argv) != 3:
        print("Usage: " + sys.argv[0] + " format failed")
        print("[python HeuristicClassifier.py keyword.txt comments.csv]")
        exit(-2)

    # Grab current time before running the code
    start = time.time()

    #Get the input file -- arg1 -- keywords
    keyword_file = sys.argv[1]
    keywords_set = open_mapping_file(keyword_file)
    #print (keywords_set)

    # Process Pull Requests (PRs) comments
    comment_file = sys.argv[2]
    print("Processing.", end ="\n")
    process_pull_requests(keywords_set, comment_file)
    print("\nDone")

    # Grab current time after running the code
    end = time.time()

    total_time = end - start

    print("\n Execution Time: " + str(datetime.timedelta(seconds=total_time)))