import ijson
import logging
import datetime
import os.path
import numpy as np
from evaluator import Evaluator
from metrics import Metrics
from filewriter import FileWriter
from index import Index
from math import sqrt
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer

test_set_path = "./testSet"
training_set_path = "./trainingSet"
topics = 100
passes = 1
test_set_limit = 200
threshold_start = 1
threshold_end = 10

thresholds = []
metrics_obj_list = []

fw = FileWriter()

for i in range(threshold_start, threshold_end+1):
    thresholds.append(i)
    metrics_obj_list.append(Metrics())

#Tokenizer
tokenizer = RegexpTokenizer(r'\w+')
#Stop word list
en_stop = get_stop_words('en')
#Porter Stemmer
p_stemmer = PorterStemmer()

logging.basicConfig(level = logging.ERROR)
logger = logging.getLogger(__name__)

def getTime():
    return str(datetime.datetime.time(datetime.datetime.now()))

def tlog(msg):
    print("["+getTime()+"] "+msg)

def processDoc(doc):
    text = doc["title"] + " " + doc["abstractText"]

    # clean and tokenize document string
    raw = text.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]

    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

    # add tokens to list
    doc["abstractText"] = stemmed_tokens

    return doc

start_time = getTime()

#Reading both sets
training_set = open(training_set_path, "r", encoding="ISO-8859-1")
test_set = open(test_set_path, "r")

#Processing both sets - tokenization, stop-word removal, stemming
tlog("Processing training set.")
processed_train_docs = []
training_set_docs = ijson.items(training_set, "documents.item")
for doc in training_set_docs:
    processed_train_docs.append(processDoc(doc))

tlog("Training set processed.")

tlog("Processing test set.")
processed_test_docs = []
test_set_docs = ijson.items(test_set, "documents.item")
i = 0
for doc in test_set_docs:
    if i == test_set_limit:
        break

    processed_test_docs.append(processDoc(doc))

    i+=1

tlog("Test set processed.")
#Query and evaluate results

eval = Evaluator()

tlog("Creating topic index.")
index = Index(processed_train_docs)
tlog("Topic index done.")

for i in range(0, len(processed_test_docs)):
    '''
    cos_results = []
    for j in range(0,len(training_set)):
        cos_sim = cossim([y for (x,y) in train_topic_list[j]], [y for (x,y) in query_doc_topics])
        cos_results.append((cos_sim, j))

    tlog("Sorting by similarity score.")
    cos_results.sort(key=lambda tup: tup[0], reverse=True)

    #fw.writeQueryResults(results[0:thresholds[-1]], i)
    '''

    results = index.query(processed_test_docs[i])

    for k in range(0, len(thresholds)):
        threshold = thresholds[k]

        eval.query([processed_train_docs[x] for (x, y) in results[0:threshold]], processed_test_docs[i])
        eval.calculate()

        #metrics_obj_list[k].updateConfusionMatrix(eval)
        metrics_obj_list[k].updateMacroAverages(eval)


for obj in metrics_obj_list:
    obj.calculate(len(processed_test_docs))

fw.writeToFiles(metrics_obj_list, thresholds)

tlog("Done.")

end_time = getTime()

print("Start time: " + start_time)
print("End time: " + end_time)
