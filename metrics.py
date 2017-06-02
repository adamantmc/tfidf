import math

class Metrics:
    def __init__(self):
        self.average_doc_precision = 0
        self.average_doc_recall = 0
        self.average_doc_f1score = 0

        self.doc_precision_values = []
        self.doc_recall_values = []
        self.doc_f1score_values = []

    def updateMacroAverages(self, eval):
        self.doc_precision_values.append(eval.getAverageDocPrecision())
        self.doc_recall_values.append(eval.getAverageDocRecall())
        self.doc_f1score_values.append(eval.getAverageDocF1score())

    def calculate(self, test_set_size):
        self.average_doc_precision = 0
        self.average_doc_recall = 0
        self.average_doc_f1score = 0

        prec_variance = 0
        rec_variance = 0
        f1_variance = 0

        for i in range(test_set_size):
            self.average_doc_precision += self.doc_precision_values[i]
            self.average_doc_recall += self.doc_recall_values[i]
            self.average_doc_f1score += self.doc_f1score_values[i]

        self.average_doc_precision = self.average_doc_precision / test_set_size
        self.average_doc_recall = self.average_doc_recall / test_set_size
        self.average_doc_f1score = self.average_doc_f1score / test_set_size

        for i in range(test_set_size):
            prec_variance += math.pow(self.doc_precision_values[i] - self.average_doc_precision, 2)
            rec_variance += math.pow(self.doc_recall_values[i] - self.average_doc_recall, 2)
            f1_variance += math.pow(self.doc_f1score_values[i] - self.average_doc_f1score, 2)

        prec_variance /= test_set_size
        rec_variance /= test_set_size
        f1_variance /= test_set_size

        self.doc_precision_std_dev = math.sqrt(prec_variance)
        self.doc_recall_std_dev = math.sqrt(rec_variance)
        self.doc_f1score_std_dev = math.sqrt(f1_variance)
