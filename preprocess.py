import nltk
import re
import math
from collections import defaultdict


# Constants
WORD_REGEX = '[a-z]+'
NORMAL_WEIGHT = 1
TITLE_WEIGHT = 5
TOPIC_WEIGHT = 1
PLACE_WEIGHT = 1
THRESHOLD_PERCENTAGE = 0.75

stemmer = nltk.stem.porter.PorterStemmer()
#lem = nltk.stem.wordnet.WordNetLemmatizer()



#############
# Functions #
#############

# Strip/replace specific characters
def stripchars(string):
    string = re.sub( '[<>]', '', string) #remove <, >
    string = string.replace('\n',' ') #remove newlines    
    return string

# Return dictionary specifying the word count in the title and text of article
def get_frequency(record):
    freq_dict = defaultdict(int)

    title = record.get("title", "default")
    title = stripchars(title)
    for word in re.findall(WORD_REGEX, title):
        #word = lem.lemmatize(word)
        word = stemmer.stem(word)
        freq_dict[word] += TITLE_WEIGHT

    text = record.get("text", "default")
    text = stripchars(text)
    for word in re.findall(WORD_REGEX, text):
        #word = lem.lemmatize(word)
        word = stemmer.stem(word)
        freq_dict[word] += NORMAL_WEIGHT

    return freq_dict


# sorted_tuple_list is a list of tuples
# Find the index of tuple from which to trim
def find_index(sorted_tuple_list, value):
    for i, v in enumerate(sorted_tuple_list):
        if v[1] > value:
            return i
    return -1        


# Compute and write Document Frequency and Inverse Document Frequency to file
def write_IDF(document_freq_dict_sorted):
    print "Writing inverse document frequency to IDF.csv"
    total_docs= float(len(document_freq_dict_sorted))
    idf_file = open("IDF.csv","w")
    idf_file.write("Word, Document Frequency, Inverse Document Frequency\n")
    for i in document_freq_dict_sorted:
        idf_file.write(i[0] + "," + str(i[1]) + "," + str(math.log(total_docs/i[1])) + "\n")
    idf_file.close()


# Find the thresholds on sorted Document Frequency and trim based on it
def get_trimmed_list(document_freq_dict_sorted):
    max_freq = document_freq_dict_sorted[-1][1]
    lower_threshold = max_freq/100*THRESHOLD_PERCENTAGE
    upper_threshold = max_freq/100*(100 - THRESHOLD_PERCENTAGE)

    lower_index = find_index(document_freq_dict_sorted, lower_threshold)
    upper_index = find_index(document_freq_dict_sorted, upper_threshold)
    trimmed_list = document_freq_dict_sorted[lower_index:upper_index]
    return trimmed_list


# Remove stopwords
def remove_stopwords(trimmed_list, stopwords):
    word_list = []
    for i in trimmed_list:
        if not i[0] in stopwords:
            word_list.append(i[0])
    return word_list


# Write the word list to word_list.txt
def write_word_list(word_list):
    print "Writing word list in file word_list.txt"
    word_file = open('word_list.txt','w')
    for word in word_list:
        word_file.write(word + '\n')
    word_file.close()


# Create all topics list
def create_topics_list(article_data_list):
    topics_set = set()
    for record in article_data_list:
        topics_set.update(record.get("topics", []))
    return list(topics_set)


# Create all places list
def create_places_list(article_data_list):
    places_set = set()
    for record in article_data_list:
        places_set.update(record.get("places", []))
    return list(places_set)


# Write data matrix to data_matrix.csv
def write_data_matrix(article_data_list, word_list, topics_list, places_list):
    dmat_file = open("data_matrix.csv", "w")

    # on the first line, write word# / topic# / place#
    for index in range(1, 1+len(word_list)):
        dmat_file.write(", Word " + str(index))
    for index in range(1, 1+len(topics_list)):
        dmat_file.write(", Topic " + str(index))
    for index in range(1, 1+len(places_list)):
        dmat_file.write(", Place " + str(index))
    dmat_file.write("\n")

    # On the second line, write actual words/topics/place names
    dmat_file.write("Article #")
    for word in word_list:
        dmat_file.write("," + word)
    for topic in topics_list:
        dmat_file.write("," + "t_"+topic)
    for place in places_list:
        dmat_file.write("," + "p_"+place)
    dmat_file.write("\n")

    # Each line is for an article
    for article_data in article_data_list:
        string = "Article " + str(article_data["article_id"])
        for word in word_list:
            string += "," + str(article_data["freq_dict"][word])
        for topic in topics_list:
            if topic in article_data["topics"]:
                string += "," + str(TOPIC_WEIGHT)
            else:
                string += ",0"
        for topic in places_list:
            if topic in article_data["places"]:
                string += "," + str(PLACE_WEIGHT)
            else:
                string += ",0"
        dmat_file.write(string + "\n")
    dmat_file.close()


# Write transaction matrix to transaction_matrix.csv
def write_transaction_matrix(article_data_list, word_list):
    tmat_file = open("transaction_matrix.csv", "w")
    tmat_file.write("Article Id, Bag of words, Bag of Topics, Bag of Places\n")
    for article_data in article_data_list:
        string = "Article " + str(article_data["article_id"])
        bag = []
        for word in word_list:
            if article_data["freq_dict"][word] > 0:
                bag.append(word)
        string += ",\"" + ", ".join(bag) + "\""
        string += ",\"" + ", ".join(article_data["topics"]) + "\""
        string += ",\"" + ", ".join(article_data["places"]) + "\""
        tmat_file.write(string + "\n")
    tmat_file.close()


