class Evaluator:
    def __init__(self):
        pass

    def query(self, retrieved_docs, query_doc):
        self.retrieved_docs = retrieved_docs
        self.query_doc = query_doc

    def calculate(self):
        self.average_doc_precision = 0
        self.average_doc_recall = 0
        self.average_doc_f1score = 0
        self.average_precision = 0

        retrieved_labels = set()
        prev_recall = 0

        doc_tp = 0
        doc_fp = 0
        doc_fn = 0

        for doc in self.retrieved_docs:
            relevant = 0

            t_doc_tp = 0
            t_doc_fp = 0
            t_doc_fn = 0

            for label in doc["meshMajor"]:
                retrieved_labels.add(label)
                if label in self.query_doc["meshMajor"]:
                    t_doc_tp += 1
                else:
                    t_doc_fp += 1

            doc_tp += t_doc_tp
            doc_fp += t_doc_fp
            doc_fn += len(self.query_doc["meshMajor"]) - t_doc_tp

            map_relevant = 0
            for label in retrieved_labels:
                if label in self.query_doc["meshMajor"]:
                    map_relevant += 1

            prec_at_i = map_relevant / len(retrieved_labels)
            rec_at_i = map_relevant / len(self.query_doc["meshMajor"])

            self.average_precision += prec_at_i*(rec_at_i - prev_recall)
            prev_recall = rec_at_i

        self.average_doc_precision = doc_tp / (doc_tp + doc_fp)
        self.average_doc_recall = doc_tp / (doc_tp + doc_fn)

        if self.average_doc_precision + self.average_doc_recall != 0:
            self.average_doc_f1score = 2*(self.average_doc_precision*self.average_doc_recall) / (self.average_doc_precision + self.average_doc_recall)
        else:
            self.average_doc_f1score = 0

    def getAverageDocPrecision(self):
        return self.average_doc_precision

    def getAverageDocRecall(self):
        return self.average_doc_recall

    def getAverageDocF1score(self):
        return self.average_doc_f1score

    def getAveragePrecision(self):
        return self.average_precision

    def printResults(self):
        print("Doc average precision: " + str(self.getDocAveragePrecision()))
        print("Doc average recall: " + str(self.getDocAverageRecall()))
        print("Doc average f1score: " + str(self.getDocAverageF1score()))
