import csv
import os
import sys
from concurrent.futures.thread import ThreadPoolExecutor
from os import listdir
from os.path import isfile, join
from time import time
from word2keypress import Keyboard

kb = Keyboard()


# The function is_valid_password filters password in this way:
# - removes passwords longer than 30 characters or shorter than 4.
# - removes non ASCII printable characters in a password
# - removes bot, which are recognisable by the same mail used more than 100 times.
# - removes HEX passwords (identified by $HEX[]) and \x
# - removes HTML char set

def is_valid_password(password):
    pass_len = len(password)
    return 4 < pass_len < 30 and password.isascii() and password.isprintable() and not password.startswith('\\x') \
           and '$HEX' not in password and '&lt' not in password and '&gt' not in password \
           and '&le' not in password and '&ge' not in password and '&#' not in password \
           and '&amp' not in password


# The function filter_file filters source, removing mail in this way:
# - mail which appear more than 100 times and less than 2.
# - mail with non-valid password

# After that the password is converted in a key-press sequence.
# The result will be saved in dest, which is a csv file, in this way:

# sample@gmail.com: ["'97314348'", "'voyager1'"]

def filter_file(source, dest):
    # Counter of occurrences of the same mail
    counter = 0
    # temp_list will contain the new line read from source, which will be printed only if
    # the email address is not repeated more than 100 times and less than 2 times.
    # After the print, the elements in the list will be deleted.
    temp_list = []
    # Opening a new csv file with the filtered file
    with open(dest, mode='w') as new_file:
        new_file_writer = csv.writer(new_file, delimiter=':', quoting=csv.QUOTE_NONE, escapechar='', quotechar='')
        # In order to avoid problems and Decode
        with open(source, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                split_line = line.strip().split(':')
                # If line is valid, continue processing
                if len(split_line) == 2:
                    # Retrieve mail and password
                    (email, password) = split_line
                    # Check if current mail is different from the last one read
                    if len(temp_list) > 0 and email != temp_list[-1][0]:
                        # Check counter: if true, write elements to CSV
                        if 2 <= counter < 100:
                            email_list, password_list = zip(*temp_list)
                            password_list = [f'{repr(kb.print_keyseq(kb.word_to_keyseq(p)))}' for p in
                                             password_list]  # to key-presses + double ticks
                            password_list_str = f"[{', '.join(password_list)}]"
                            # Write all elements of the list to file
                            new_file_writer.writerow([email_list[0], password_list_str])
                        counter = 0
                        temp_list.clear()
                    # If password is valid, add new tuple to list
                    if is_valid_password(password):
                        # Add tuple (email, password) to list and increment counter
                        temp_list.append(split_line)
                        counter += 1

        # The last item or the last items with the same email cannot be printed.
        # So the last elements will be printed and the list will be cleared
        if 2 <= counter < 100:
            email_list, password_list = zip(*temp_list)
            password_list = [f'\"{repr(kb.print_keyseq(kb.word_to_keyseq(p)))}\"' for p in
                             password_list]  # to key-presses + double ticks
            password_list_str = f"[{', '.join(password_list)}]"
            # print(email_list[0], password_list_str)
            # Write all elements of the list to file
            new_file_writer.writerow([email_list[0], password_list_str])
        temp_list.clear()


if __name__ == '__main__':
    start_time = time()
    arguments = sys.argv
    if len(arguments) != 3:
        print('Usage: python3 filter.py <dir_source_path> <dir_dest_path>')
        exit(1)
    source_dir = arguments[1]
    dest_dir = arguments[2]
    # if not isdir(source_dir):
    #    print("error, source directory not found, exiting")
    #    exit(2)
    for file in listdir(source_dir):
        source_file = join(source_dir, file)
        if isfile(source_file):
            if not os.path.isdir(dest_dir):
                os.makedirs(dest_dir)
            # Creates the file filtered in the specified path and with the same name as filename.
            dest_file = os.path.join(dest_dir, file + ".csv")
            with ThreadPoolExecutor(max_workers=4) as executor:
                executor.submit(filter_file, source_file, dest_file)
        stop_time = time()
        print(f'Time elapsed: {stop_time - start_time} sec')
