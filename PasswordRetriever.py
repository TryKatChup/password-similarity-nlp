import csv
import time

# This class retrieves a list of passwords from the cleaned csv file.
class PasswordRetriever(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename) as file:
            csv_reader = csv.reader(file, delimiter=':')
            start_time = time.time()
            for i, (user, pass_keyseq_list) in enumerate(csv_reader):
                if i % 1000000 == 0:
                    end_time = time.time()
                    print("Processed {} lines in {} seconds.".format(i, end_time - start_time))
                    start_time = end_time
                user_pass_list = eval(pass_keyseq_list)
                yield user_pass_list
