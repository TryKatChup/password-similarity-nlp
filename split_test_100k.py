import sys
import csv
import random


def main():
    args = sys.argv
    if len(args) != 3:
        print("Usage: python split_dataset.py SOURCE_FILE TEST_DEST_FILE")
        sys.exit(-1)

    source = args[1]
    test_dest = args[2]
    test_chance = 0.01  # IMPORTANT: 0.1 -> 10% of the whole dataset reserved to test

    random.seed(42)  # IMPORTANT: fix seed to have reproducible results
    with open(source, mode='r') as s_f, open(test_dest, mode='w') as test_f:
        source_csv_reader = csv.reader(s_f, delimiter=':')
        test_csv_writer = csv.writer(test_f, delimiter=':', quoting=csv.QUOTE_NONE, quotechar='', escapechar='')

        for i, (user, pass_keyseq_list) in enumerate(source_csv_reader):
           #  print(f'Processing line {i+1}...')
            # Apply reservoir sampling
            if random.random() < test_chance:
                test_csv_writer.writerow([user, pass_keyseq_list])
    print('Processing complete!')


if __name__ == "__main__":
    main()


