import sys
import csv
import random


def main():
    args = sys.argv
    if len(args) != 4:
        print("Usage: python split_dataset.py SOURCE_FILE TRAIN_DEST_FILE TEST_DEST_FILE")
        sys.exit(-1)

    source = args[1]
    train_dest = args[2]
    test_dest = args[3]
    test_chance = 0.1  # IMPORTANT: 0.1 -> 10% of the whole dataset reserved to test

    random.seed(42)  # IMPORTANT: fix seed to have reproducible results
    with open(source, mode='r') as s_f, open(train_dest, mode='w') as train_f, open(test_dest, mode='w') as test_f:
        source_csv_reader = csv.reader(s_f, delimiter=':')
        train_csv_writer = csv.writer(train_f, delimiter=':', quoting=csv.QUOTE_NONE, quotechar='', escapechar='')
        test_csv_writer = csv.writer(test_f, delimiter=':', quoting=csv.QUOTE_NONE, quotechar='', escapechar='')

        for i, (user, pass_keyseq_list) in enumerate(source_csv_reader):
           #  print(f'Processing line {i+1}...')
            # Apply reservoir sampling
            if random.random() > test_chance:
                train_csv_writer.writerow([user, pass_keyseq_list])
            else:
                test_csv_writer.writerow([user, pass_keyseq_list])
    print('Processing complete!')


if __name__ == "__main__":
    main()
