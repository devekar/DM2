#!/usr/bin/python


#  Algorithm:
#  1. Maintain a document frequency hash.
#  2. Iterate through file, strip invalid characters and feed it to ArticleParser.
#  3. ArticleParser returns list of article data(id, topics, places, title, text) in that file.
#  4. For each article, apply stemming on each word.
#     Build word-count for title and text; also update document frequency.
#  5. Sort the document frequency by the value(frequency) and trim 0.75% on value.
#  6. Load list of stopwords from file and remove stopwords from trimmed words.
#  7. Create topics and places lists from all articles.
#  8. Write data matrix and transaction matrix to files.

import os
import re
import timeit
import operator
from collections import defaultdict

from parse import ArticleParser
from preprocess import *

start = timeit.default_timer()

# Constants
PATH = r'reuters'



########
# Main #
########

# article_data_list is a list, with each element a dictionary.
# Store the word count for each article in this list.
# The dictionary(for an aricle) consists of:
# 1. article_id - The NEWID of reuters article
# 2. topics - This is a list of topics
# 3. places - This is a list of places
# 4. freq_dict - This is again a dictionary consisting of word-count pairs
article_data_list = []  
document_freq_dict = defaultdict(int)   # Store the number of articles in which a word appears

# Iterate through each file, parse and iterate through article
parse_time = -timeit.default_timer()
for dir_entry in sorted(os.listdir(PATH)):   #each file
    dir_entry_path = os.path.join(PATH, dir_entry)
    file = open(dir_entry_path)
    print "Reading file: " + dir_entry_path
    content = re.sub('&(.+?);|,|\'|"', '',file.read())  #remove &xxx; and comma and quotes, which may interfere with parsing
    parser = ArticleParser()  # Used for parsing the files
    parser.feed(content)
    file.close()

    for record in parser.records_list: #each article
        article_id = record.get("article_id", -1)
        topics_list = record.get("topics", [])
        places_list = record.get("places", [])
        freq_dict = get_frequency(record)
        article_data_list.append( {'article_id':article_id, 'topics':topics_list, 'places':places_list,'freq_dict':freq_dict} )

        for word in freq_dict.keys():
            document_freq_dict[word] += 1


parse_time += timeit.default_timer()
print "Parsed all articles in "+ str(round(parse_time,2)) + " seconds\n"


# Sort the document_freq_dict dictionary by the value
document_freq_dict_sorted = sorted(document_freq_dict.iteritems(), key=operator.itemgetter(1))

# Compute and write Document Frequency and Inverse Document Frequency to IDF.csv
write_IDF(document_freq_dict_sorted)

# Find the thresholds on IDF and trim it
trimmed_list = get_trimmed_list(document_freq_dict_sorted)
print "Trimmed word list... Number of words:  " + str(len(trimmed_list))

# Load stopwords
file = open('stopwords')
stopwords = file.read().split()
file.close()
# Remove stopwords
word_list = remove_stopwords(trimmed_list, stopwords)

# Write the word list to word_list.txt
print "Removed stopwords... Number of words: " + str(len(word_list))
write_word_list(word_list)

# Create topics and places lists
topics_list = create_topics_list(article_data_list)
places_list = create_places_list(article_data_list)
print "Total number of topics: " + str(len(topics_list))
print "Total number of places: " + str(len(places_list))

# Write data matrix to data_matrix.csv
print "Writing data matrix in file data_matrix.csv"
dm_write_time = -timeit.default_timer()
write_data_matrix(article_data_list, word_list, topics_list, places_list)
dm_write_time += timeit.default_timer()
print "Data matrix written in " + str(round(dm_write_time,2)) + " seconds"

# Write transaction matrix to transaction_matrix.csv
print "Writing transaction matrix in file transaction_matrix.csv"
trm_write_time = -timeit.default_timer()
write_transaction_matrix(article_data_list, word_list)
trm_write_time += timeit.default_timer()
print "Transaction matrix written in " + str(round(trm_write_time,2)) + " seconds"

end = timeit.default_timer()
print "Total Execution Time :" + str(round(end- start,2))


