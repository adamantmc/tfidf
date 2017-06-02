import operator
import math

class Index:
    def __init__(self, documents):
        self.index = {}
        self.document_norms = {}
        self.idfs = {}

        for doc_id in range(len(documents)):
            self.document_norms[doc_id] = 0

            for word in documents[doc_id]["abstractText"]:
                #Each topic is a tuple: (topic_id, probability)
                if word not in self.index:
                    self.index[word] = {}

                if doc_id not in self.index[word]:
                    self.index[word][doc_id] = 0

                self.index[word][doc_id] = self.index[word][doc_id] + 1

        for word in self.index:
            idf = 1 + math.log(len(documents) / len(self.index[word]))
            self.idfs[word] = idf
            for doc in self.index[word]:
                tf = self.index[word][doc] / len(documents[doc]["abstractText"])
                self.index[word][doc] = tf*idf
                self.document_norms[doc] = self.document_norms[doc] + math.pow(self.index[word][doc], 2)

        for doc in range(len(documents)):
            self.document_norms[doc] = math.sqrt(self.document_norms[doc])

    def query(self, query_doc):
        query_index = {}
        for word in query_doc["abstractText"]:
            if word not in query_index:
                query_index[word] = 0
            query_index[word] = query_index[word] + 1

        scores = {}

        for word in query_doc["abstractText"]:
            tf = query_index[word] / len(query_doc["abstractText"])

            if word in self.index:
                for doc in self.index[word]:
                    idf = self.idfs[word]
                    if doc not in scores:
                        scores[doc] = 0
                    scores[doc] += (tf*idf)*self.index[word][doc]

        #We ignore the norm of the query doc as it is the same for all queries
        for doc in scores:
            scores[doc] = scores[doc] / self.document_norms[doc]

        scores_sorted = sorted(scores.items(), key = operator.itemgetter(1), reverse = True)

        return scores_sorted
