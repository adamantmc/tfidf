import os

class FileWriter:
    def __init__(self, dir = "results"):
        self.dir = dir
        if not os.path.exists(dir):
            os.makedirs(dir)

        self.filenames = [self.dir+"/doc_average_precision.txt",
                          self.dir+"/doc_average_recall.txt",
                          self.dir+"/doc_average_f1score.txt"]

    def writeToFiles(self, metrics_obj_list, thresholds):
        files = []
        for file in self.filenames:
            files.append(open(file, 'w'))

        for i in range(0, len(thresholds)):
            files[0].write(str(thresholds[i]) + " " + str(metrics_obj_list[i].average_doc_precision) + " " + str(metrics_obj_list[i].doc_precision_std_dev) + "\n")
            files[1].write(str(thresholds[i]) + " " + str(metrics_obj_list[i].average_doc_recall) + " " + str(metrics_obj_list[i].doc_recall_std_dev) + "\n")
            files[2].write(str(thresholds[i]) + " " + str(metrics_obj_list[i].average_doc_f1score) + " " + str(metrics_obj_list[i].doc_f1score_std_dev) + "\n")

        for file in files:
            file.close()
