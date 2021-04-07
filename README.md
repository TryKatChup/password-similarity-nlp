# Password Similarity Detection Using Deep Neural Networks
---
![](images/logodef3.png)

### Introduction

Nowadays people tend to create more profiles or change password of their current profiles for security reasons. Existent passwords and literature-based words have a great impact on the candidate password. This could be a risk for the user privacy. For example, an user has the password `mum77` and he/she wants to create a new account for a different website. A candidate password could be `mommy1977`, which is a variation of `mum77` and it more risky if an attacker has discovered the first password in a leak. 

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

![](images/US_keyboard_layout.png)

### Splitting dataset
#### File: `split_dataset.py`
Both datasets are splitted in training set (which is 90% of the original dataset) and test set (the remaining 10% of the original dataset). 

### Training FastText
#### File: `Training dataset.py` and `PasswordRetriever.py`
In this file FastText will be trained based on the given training set.
In order to understand better the file and FastText, Word2Vec is briefly introduced.

#### Word2Vec
Word2Vec is a set of architectural and optimization models which learn word embeddings from a large dataset, using deep neural networks.
A model trained with Word2Vec can detect similar words (based on context) thanks to cosine similarity.

Word2Vec is based on two architectures:
- **CBOW** (continuous bag of words): the main purpose is to combine the representation of surrounding words, in order to predict the word in the middle.
- **Skip-gram:** similar to CBOW, except for the word in the middle, which is used to predict words related to the same context.
  
CBOW is more faster and effective with larger dataset, however, despite the greater complexity, Skip-gram is capable to find _out of dictionary_ words for smaller datasets.

![](./images/cbow_vs_skipgram.png)


#### FastText
FastText is a open source library created by Facebook which extends Word2Vec and is capable to learn word representation and sentence classification efficiently. The training is based on password n-grams. 
N-grams of a specific word that contains c_1...c_m characters are defined as follow:

*formula con gli ngram*

For example, the ngrams of the word `world`, with n_mingram = 1 and n_maxram = 5 are:
`world = {{w, o, r, l, d}, {wo, or, rl, ld}, {wor, orl, rld}, {world}}`
`world` is represented as the subset of substrings with 1 and 5 as respectively minimum and maximum length.
FastText, comparing to Word2Vec, is capable to obtain more _out of dictionary_ words, which are unknown during the training phase.

#### Environment setup
In order to train the model, `gensim.FastText` was used.
`gensim` is a open source and cross-platform python library, with multiple pre-trained word embedding models. 
`PasswordRetriever` class extracts all the passwords from the `.csv` preprocessed file.

#### FastText parameters
```
negative = 5
subsampling = 1e-33
min_count = 10
min_n = 2
max_n = 4
SIZE = 200
sg = 1
```
In this project _Skip-gram_ model (`sg = 1`) and negative sampling were used:
  
- Skip-gram approach is chosen, as the distributed representation of the input word is used to predict the context. Skip-gram model works better with subword information, so it is recommended for learning passwords and rare words in general.

- Negative sampling makes the training faster. Each training sample updates only a small percentage of the model weights. For larger datasets (like this case) it is recommended to set negative sampling between 2 and 5.

- The dimension of vectors is set to 200, in order to have train the model faster. Generally it is recommended to have `SIZE = 300`.

- The `subsampling` ignores the most frequent password (more than 1000 occurrences).

- `min_count` represents the minimum number of occurences of a password in the training dataset.

- `min_n` and `max_n` are the number of respectively minimum and maximum n-grams.

- N-grams are used in statistical natural language processing to predict a word and/or the context of a word. In this case they represent a contiguous sequence of n characters and their purpose is to give subword information.

  - For example a password `w = qwerty` with `min_n = 4` and `m_max = 5`, will have the following n-grams.    
  `zw = {<qwe, qwer, wert, erty, rty>, <qwer, qwert, werty, erty>}`

  NB `<` and `>` are considered as characters.

#### Saving the model
Finally, the trained model is saved as `.bin`.
```
from gensim.models.fasttext import save_facebook_model
save_facebook_model(trained_model, "model_password_similarity.bin")
print("Model saved successfully.")
```

#### Compressing the model

The trained model has 4.8GB. There are some problems about the size:

- Too much space occupied in memory.
- It is harder to use the model in client/server architectures. Everyone should use this model, which can be sent as a payload. Embeddings are not reversible, and they guarantee password anonymization.

For these reasons, `compress_fasttext` is used.
```
import gensim
import compress_fasttext
```

In order to obtain a compressed model, without impacting significantly on performances, product quantitation and feature selection are applied.

- Feature selection is the process of selecting a subset of relevant features for use in model construction or in other words, the selection of the most important features.

- Product quantization is a particolar type of vector quantization, a lossy compression technique used in speech and image coding. A product quantizer can generate anexponentially large codebook at very low memory/time cost.

```
big_model = gensim.models.fasttext.load_facebook_vectors('model_password_similarity.bin')
small_model = compress_fasttext.prune_ft_freq(big_model, pq=True)
small_model.save('compressed_model')
```
The compressed_model is 20MB.

### Evaluating the model
#### File: `w2kp_PRGraph.py` and `No_w2kp_PRGraph.py`

For the evaluation of the models, compressed versions obtained with product quantization were used. In order to measure any performance differences between the original model and the compressed version, Bijeeta et al. model is chosen, with the following features:
- translation of the sequence of key pressed for the password;
- `min_gram = 1`;
- `max_gram = 1`;
- `epochs = 5`.

An effective valutation of both model is based on _precision_ and _recall_. 
Not remarkable differences were observed: for this reason only the compressed version of the models are considered.

**inserisci grafici qui**

#### Euristhics
For a proper evaluation the following euristhics is adopted:
- comparing the password to the lowercase version;
- comparing the password to the uppercase version;
- comparing the password to the `l33t code` version;
- verifing if edit distance is greater than 5.

#### Ground truth and prediction



#### Precision and recall



### Graphic representation of words distance
#### File: `visualize_embeddings.py`

To simplify the comprehension of the project topic, password similarity is represented with a 3-dimensional graphic. `t-SNE` algorithm is used to reduce the model dimension from 200 to 3. In the next figure it is possible to see the top 5 most similar passwords to `ipwnedyou` and `numBerOne` and their distances. 
