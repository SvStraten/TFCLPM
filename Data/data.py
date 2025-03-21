import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from edbn.Utils.LogFile import LogFile

class Data:
    def __init__(self, name, logfile):
        self.name = name
        self.logfile = logfile

        self.train = None
        self.test = None
        self.test_orig = None

        self.folds = None

    def __str__(self):
        return "Data: %s" % self.name

    def prepare(self, setting):
        print("PREPARE")
        prefix_size = max(self.logfile.get_cases().size())
        self.logfile.k = prefix_size

        if setting.add_end:
            self.logfile.add_end_events()

        if setting.filter_cases:
            self.logfile.filter_case_length(setting.filter_cases)

        print("CONVERT")
        self.logfile.convert2int()
        print("K-CONTEXT")
        self.logfile.create_k_context()
        self.logfile.contextdata = self.logfile.contextdata

        print("SPLIT TRAIN-TEST")
        if setting.train_split != "k-fold":
            # Perform train-test split here
            self.train, self.test_orig = self.logfile.splitTrainTest(
                train_percentage=setting.train_percentage,
                split_case=setting.split_cases,
                method=setting.train_split
            )
            self.train.contextdata = self.train.contextdata
            self.test_orig.contextdata = self.test_orig.contextdata
        else:
            # Handle k-fold splitting
            self.create_folds(setting.train_k)


    def create_batch(self, split="normal", timeformat=None):
        if split == "normal":
            self.test = {"full": {"data": self.test_orig}}
        elif split == "day":
            self.test = self.test_orig.split_days(timeformat)
        elif split == "week":
            self.test = self.test_orig.split_weeks(timeformat)
        elif split == "months":
            self.test = self.test_orig.split_months(timeformat)

        # Debugging: Check if self.test is None
        if self.test is None:
            print("Error: self.test is None after batch creation!")

        return self.test  # Ensure self.test is properly set


    def create_folds(self, k):
        self.folds = self.logfile.create_folds(k)

    def get_fold(self, i):
        self.test = self.folds[i]
        self.train = None
        for j in range(len(self.folds)):
            if i != j:
                if self.train is not None:
                    self.train = self.train.extend_data(self.folds[j])
                else:
                    self.train = self.folds[j]
                    # self.train = copy.deepcopy(self.folds[j])


    def get_batch_ids(self):
        return sorted(self.test.keys())

    def get_test_batch(self, idx):
        return self.test[self.get_batch_ids()[idx]]["data"]

    def get_test_batchi(self, idx1, idx2):
        test = self.train
        test_logfile = LogFile(None, None, None, None, test.time, test.trace, test.activity, test.values, False, False)
        test_logfile.filename = test.filename
        test_logfile.values = test.values
        # print("Index 1 is {}".format(idx1))
        # print("Index 2 is {}".format(idx2))
        # print("Size test data is {}".format(test.contextdata.shape))
        test_logfile.contextdata = test.contextdata.iloc[idx1:idx2, :]
        # print(type(test_logfile.contextdata))
        test_logfile.categoricalAttributes = test.categoricalAttributes
        test_logfile.numericalAttributes = test.numericalAttributes
        test_logfile.data = test.data.iloc[idx1:idx2, :]
        test_logfile.k = test.k
        # print(test_logfile.data.shape)
        # print(test_logfile.contextdata.shape)
        return test_logfile

    def get_batch_timestamp(self, idx):
        return self.get_batch_ids()[idx]

