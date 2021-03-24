# Password Similarity Detection Using Deep Neural Networks
---
### Introduction

Nowadays people tend to create more profiles or change password of their current profiles for security reasons. Existent passwords and literature-based words have a great impact on the candidate password. This could be a risk for the user privacy. For example, an user has the password `mum77` and he/she wants to create a new account for a different website. A candidate password could be `mommy1977`, which is a variation of `mum77`and it more risky if an attacker has discovered the first password in a leak. 

The purpose of this project is to give a feedback about password similarity between the new password and the old one using Deep Neural Networks. 
As a reference, a Scientific Article pulished at IEEE Symposium on Security and Privacy 2019 was chosen. Then the entire architecture was reimplemented and improved, and a comparison between the obtained results and the case study was made.

### Data pre-processing
#### File: `preparing_dataset.py`

First of all I used a compilation of password leaks containing 1.4 billion email-password pairs from the Deep Web. Further operations were applied on the dataset:
- Passwords longer than 30 characters or shorter than 4 removal.
- Non ASCII printable characters in password removal.
- Bot (which are recognisable by the same mail used more than 100 times) removal.
- HEX passwords (identified by `$HEX[]`) and `\x` removal.
- HTML char set removal, for example:
  - `&gt`;
  - `&ge`;
  - `&lt`;
  - `&le`;
  - `&#` (HTML entity code);
  - `amp`.

- Due to the impossibility of similarity detection, accounts with less than 2 password were removed.

After that, two dataset were build:

- The first one, according with Bijeeta et alii, contains all the passwords in key-presses format (using the `word2keypress` package).
- The second one contains all the passwords as they are.
The filtered datasets were saved in 2`.csv` files in this format:
- in the first dataset: `sample@gmail.com:["’97314348’", "’voyager<s>1’"]`
- in the second dataset: `sample@gmail.com:["’97314348’", "’voyager!’"]`
 
#### Word2press

Every password in the first dataset was translated in a keypress sequence on an ANSI american keyboard:
- Every capital letter was represented by `<s>` (the `SHIFT` key) before the lowercase version.

   e.g. `Hello -> <s>hello`
- If there is a sequence of consecutive capital letters, followed by lowercase letters, the `<c>` tag (the `CAPS LOCK` key) is inserted _before_ and _after_ the sequence, which will be represented by lowercase letters.

  e.g. `Password -> <c>pass<c>word`
- If a sequence of capital letters ends at the end of the word, the `<c>` tag wil be placed before the sequence.

  e.g. ```PASSWORD -> <c>password
  passWORD -> pass<c>word```

- If a password contains ASCII 128 special characters, the `<s>` tag will be placed before the special character, which is translated as `SHIFT + <key for the specific character>`

  e.g. ```PASSWORD! -> <c>password<s>1
          Hello@!! -> <s>hello<s>2<s>1<s>1```
