---
layout: post
title: Train Word2Vec Model on Custom Corpus
subtitle: Train Word2Vec Model on Custom Corpus
cover-img: /assets/img/2021-03-02-train-word2vec-on-custom-corpus/jon-tyson.jpg
thumbnail-img: /assets/img/2021-03-02-train-word2vec-on-custom-corpus/bloomberg.jpg
readtime: true
show-avatar: false
tags: [WSL, NLP, Ubuntu]
comments: true
---

When I was doing my dissertation project, I found out that the performance of model wasn't quite well. I believe it's because the domain of pre-trained GoogleNews-vectors-negative300 is different from the the dataset of mine. Hence, I decide to pre-train a word2vec model by myself. In this article, I'll use a library called "Koan" released by Bloomberg LP. They build CBOW model using C++, which is more efficiently compared to [word2vec](https://github.com/tmikolov/word2vec/) and [gensim](https://github.com/RaRe-Technologies/gensim/) libraries. If you are using Windows, and you don't have a Linux system in your computer, please read this [article]({{site.baseurl}}{% link _posts/2021-01-22-train-word2vec-on-wsl.md %}) I wrote before to set up your WSL.

# Introduction

The reason we care about language is that, because of language, we are able to turn invisible ideas into visible actions. However, language is ambiguous at all levels: lexical, phrasal, semantic. To address this, we need to build a language model, which can convert text into vectors. The most common techniques are Bag of Words (One-Hot Encoding, TF-IDF), Distributional Word Embedding (Word2Vec, GloVe, FastText), and Contextualised Word Embedding (ELMo, BERT). In this article, I'm gonna implement Word2Vec to generate pre-trained vectors.

# Word2Vec

Word2Vec is a statistical-based method to obtain word vectors, and it is proposed by Tomas Mikolov et al. [4] of Google in 2013. Word2Vec is available in two flavors, the CBoW model and the Skip-Gram model, which is based on neural networks which can map words to low dimensional space. CBoW model predicts the current word by context, and Skip-Gram model predicts context by current word.

# Text Pre-processing

First, you need to read in your csv file containing texts.

```python
df = pd.read_csv(r"./20061020_20131126_bloomberg_news.csv")
df["title"] = df["title"].apply(str)
df["paragraph"] = df["paragraph"].apply(str)
df.sample(3)
```

--- 

|  | title | timestamp | paragraph |
| :-: | :-: | :-: | :-: |
|  6493 | Coronavirus: Malaysia's Economy Shows Doing th... | 2020/8/23 | Strict lockdowns, accommodative central banks,... |
| 1833 | Lower Rates: Trump and the Markets Picked Thei... | 2019/8/7 | Collapsing bond yields aren't exactly a sign ... |
| 4376 | Crypto Brokerage Tagomi Gets $12 Million in Se... | 2019/3/4 | Tagomi Holdings Inc., a digital asset brokerag... |

Second, put them into a list.

```python
documents = []
documents.extend(df.loc[:, ["title", "paragraph"]].values.flatten().tolist())
```

Third, do some text cleaning work.

```python
def regex(text):
    text = re.sub(r"([^a-zA-Z0-9\.\?\,\!\%\']+)", " ", text)
    text = re.sub(r"(?<=\d),(?=\d)+", "", text)
    text = re.sub(r"\,", " , ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"\!", " ! ", text)
    text = re.sub(r"\.", " . ", text)
    text = re.sub(r"  ", " ", text)
    text = text.strip()
    return text

docs = [regex(doc) for doc in documents]
docs_cased = [regex(doc.lower()) for doc in documents]
```

# Tokenisation

```python
def progressbar(iter, prefix="", size=50, file=sys.stdout):
    count = len(iter)
    def show(t):
        x = int(size*t/count)
        # file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), int(100*t/count), 100))
        file.write("{}[{}{}] {}%\r".format(prefix, "█"*x, "."*(size-x), int(100*t/count)))
        file.flush()
    show(0)
    for i, item in enumerate(iter):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()

class Tokenizer(object):
    
    def __init__(self, 
                 char_level=False, 
                 num_tokens=None, 
                 pad_token='<PAD>', 
                 oov_token='<UNK>', 
                 token_to_index=None
                ):
        self.char_level = char_level
        self.separator = '' if self.char_level else ' '
        # <PAD> + <UNK> tokens
        if num_tokens: num_tokens -= 2
        self.num_tokens = num_tokens
        self.oov_token = oov_token
        if not token_to_index:
            token_to_index = {'<PAD>': 0, '<UNK>': 1}
        self.token_to_index = token_to_index
        self.index_to_token = {v: k for k, v in self.token_to_index.items()}

    def __len__(self):
        return len(self.token_to_index)

    def __str__(self):
        return f"<Tokenizer(num_tokens={len(self)})>"

    def fit_on_texts(self, texts):
        if self.char_level:
            all_tokens = [token for text in texts for token in text]
        if not self.char_level:
            all_tokens = [token for text in texts for token in text.split(' ')]
        counts = Counter(all_tokens).most_common(self.num_tokens)
        self.min_token_freq = counts[-1][1]
        for token, count in progressbar(counts, prefix="VOCAB"):
            index = len(self)
            self.token_to_index[token] = index
            self.index_to_token[index] = token
        return self

    def texts_to_sequences(self, texts):
        sequences = []
        for text in progressbar(texts, prefix="TEXT2SEQ"):
            if not self.char_level:
                text = text.split(' ')
            sequence = []
            for token in text:
                sequence.append(self.token_to_index.get(
                    token, self.token_to_index[self.oov_token]))
            sequences.append(sequence)
        return sequences

    def sequences_to_texts(self, sequences):
        texts = []
        for sequence in progressbar(sequences, prefix="DEQ2TEXT"):
            text = []
            for index in sequence:
                text.append(self.index_to_token.get(index, self.oov_token))
            texts.append(self.separator.join([token for token in text]))
        return texts

    def save(self, fp):
        with open(fp, 'w') as fp:
            contents = {
                'char_level': self.char_level,
                'oov_token': self.oov_token,
                'token_to_index': self.token_to_index
            }
            json.dump(contents, fp, indent=4, sort_keys=False)

    @classmethod
    def load(cls, fp):
        with open(fp, 'r') as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)
```

---

```python
tokeniser = Tokenizer(char_level=False, num_tokens=1000000)
tokeniser.fit_on_texts(docs_cased[:])
sequences = tokeniser.texts_to_sequences(docs_cased[:])
texts = tokeniser.sequences_to_texts(sequences)
```

After tokenised our corpus, save it to a `news.tokens` file.

```python
with open('./news.tokens', 'w') as f:
    for item in texts:
        f.write("%s\n" % item)
```

Move your `news.tokens` file to WSL folder. In my case, it is at `C:\Users\yangwang\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu18.04onWindows_79rhkp1fndgsc\LocalState\rootfs\home\yang\`.

Next, open your mobaxterm and execute the following code.

```bash
./build/koan -V 1000000 \
             --epochs 10 \
             --dim 300 \
             --negatives 5 \
             --context-size 5 \
             -l 0.075 \
             --threads 16 \
             --cbow true \
             --min-count 2 \
             --file ./news.tokens
```

Learned embeddings will be saved to `embeddings_${CURRENT_TIMESTAMP}.txt` in the present working directory.

# Gensim

Move your pre-trained vectors back to your Windows folder, and change your file name to `news-cbow-negative300.txt`. We then convert GloVe vectors format into the word2vec format.

```python
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

_ = glove2word2vec("./news-cbow-negative300.txt", "./news-word2vec-cbow-negative300.txt")
wv_from_text = KeyedVectors.load_word2vec_format("./news-word2vec-cbow-negative300.txt", binary=False)
```

**Notes**

GloVe format (a real example can be found on the [Stanford site](https://nlp.stanford.edu/projects/glove/))

```bash
word1 0.123 0.134 0.532 0.152
word2 0.934 0.412 0.532 0.159
word3 0.334 0.241 0.324 0.188
...
word9 0.334 0.241 0.324 0.188
```

Word2Vec format (a real example can be found in the [old w2v repository](https://code.google.com/archive/p/word2vec/)).

```bash
9 4
word1 0.123 0.134 0.532 0.152
word2 0.934 0.412 0.532 0.159
word3 0.334 0.241 0.324 0.188
...
word9 0.334 0.241 0.324 0.188
```

Voilà! You have successfully got a pre-trained word embedding!

```python
wv_from_text.similar_by_word("bitcoin")
```

---

```bash
[('cryptocurrency', 0.7397603392601013),
 ('cryptocurrencies', 0.7099655866622925),
 ('crypto', 0.6509920358657837),
 ('xrp', 0.5511361360549927),
 ('ethereum', 0.547865629196167),
 ('monero', 0.5345401167869568),
 ("bitcoin's", 0.5305401086807251),
 ('bitcoins', 0.5253546237945557),
 ('gold', 0.5229815244674683),
 ('blockchain', 0.508536159992218)]
```

# Conclusion

You can now perform various syntactic/semantic NLP word tasks with the trained vectors! Cheers!