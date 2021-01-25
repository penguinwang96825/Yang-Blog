---
layout: post
title: Train Wrod2Vec Model on WSL
subtitle: Train Wrod2Vec Model on WSL
cover-img: /assets/img/2021-01-22-train-word2vec-on-wsl/michael.jpg
thumbnail-img: /assets/img/2021-01-22-train-word2vec-on-wsl/CBOW_eta_Skipgram.png
readtime: true
show-avatar: false
tags: [WSL, NLP, Ubuntu]
comments: true
---

Bloomberg LP recently published a new paper, claimed that popular implementations of word2vec with negative sampling such as [word2vec](https://github.com/tmikolov/word2vec/) and [gensim](https://github.com/RaRe-Technologies/gensim/) do not implement the CBOW update correctly, thus potentially leading to misconceptions about the performance of CBOW embeddings when trained correctly. Therefore, they release [kōan](https://github.com/bloomberg/koan) so that others can efficiently train CBOW embeddings using the corrected weight update.

In this article, I'm going to build my own pre-trained word embedding on WSL, which stands for Windows Subsystem for Linux, and it is a compatibility layer for running Linux binary executables (in ELF format) natively on Windows 10.. The reason why I train the model on Linux instead of Windows is that it's not user-freiendly to run C++ and some other packages on Windows.

# Windows-Subsystem-for-Linux

## Pre-steps
1. Download PowerShell from [here](https://github.com/PowerShell/PowerShell/releases).
2. Install Ubuntu from microsoft shop.
![ubuntu](/assets/img/2021-01-22-train-word2vec-on-wsl/ubuntu.jpg)
3. Open PowerShell and type command below.
```bash
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
```
Once you done the 3 steps above, you can visit your linux homepage through `C:\Users\<username>\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu18.04onWindows_79rhkp1fndgsc\LocalState\rootfs\home\`

## Connect with GitHub

### Generate a New SSH Key

What is `ssh-keygen`? `ssh-keygen` is a tool for creating new authentication key pairs for SSH. Such key pairs are used for automating logins, single sign-on, and for authenticating hosts. The SSH protocol uses public key cryptography for authenticating hosts and users. The authentication keys, called SSH keys, are created using the keygen program.

1. The client can generate a public-private key pair as follows: `ssh-keygen`. After this, you will see something like this.
```bash
Generating public/private rsa key pair.
Enter file in which to save the key (/home/ylo/.ssh/id_rsa): 
Enter passphrase (empty for no passphrase): 
Enter same passphrase again: 
Your identification has been saved in /home/ylo/.ssh/id_rsa.
Your public key has been saved in /home/ylo/.ssh/id_rsa.pub.
The key fingerprint is:
SHA256:Up6KjbnEV4Hgfo75YM393QdQsK3Z0aTNBz0DoirrW+c ylo@klar
The key's randomart image is:
+---[RSA 2048]----+
|    .      ..oo..|
|   . . .  . .o.X.|
|    . . o.  ..+ B|
|   .   o.o  .+ ..|
|    ..o.S   o..  |
|   . %o=      .  |
|    @.B...     . |
|   o.=. o. . .  .|
|    .oo  E. . .. |
+----[SHA256]-----+
```
2. Now, you can find your public key as follows: `cat ~/.ssh/id_rsa.pub`
3. Open your [GitHub](https://github.com) and go to the `Settings` section.
![](/assets/img/2021-01-22-train-word2vec-on-wsl/settings.png)
4. Go to `SSH and GPG keys` and click the `New SSH key` button.
5. Copy your public key to `Key` text field and press `Add SSH key`.
![](/assets/img/2021-01-22-train-word2vec-on-wsl/ssh.png)

## Ananconda for Python (Optional)
I wrote a [blog]({{site.baseurl}}{% link _posts/2020-12-31-conda-environment-setup.md %}) discussing about why we should use anaconda for python. Please take a look first.

### Install Packages
```console
sudo apt-get update
sudo apt-get install python-pip
```

### Install Anaconda
1. Download anaconda3.
```console
curl -O https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
bash Anaconda3-2020.02-Linux-x86_64.sh
export PATH=~/anaconda3/bin:$PATH
conda --version
```
2. Update and build virtual environment.
```console
conda config --set auto_activate_base false
conda update conda
conda update anaconda
conda create --name nlp python=3.7
source activate nlp
conda install ipykernel -y
python -m ipykernel install --user --name nlp --display-name "nlp"
```

**CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.**

If can not activate conda environment, I come up with a workaround below.
```console
source activate
conda deactivate
conda activate nlp
```

## Install Python Packages
```console
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch -y
```

# Train Word2Vec using Kōan

It is a common belief in the NLP community that continuous bag-of-words (CBOW) word embeddings tend to underperform skip-gram (SG) embeddings. Ozan Irsoy and Adrian Benton from Bloomberg LP show that their correct [implementation](https://arxiv.org/pdf/2012.15332.pdf) of CBOW yields word embeddings that are fully competitive with SG on various intrinsic and extrinsic tasks while being more than three times as fast to train.

## Building

To train word embeddings on Wikitext-2, first clone and build koan:
```bash
git clone --recursive git@github.com:bloomberg/koan.git
cd koan
mkdir build
cd build
cmake .. && cmake --build ./
cd ..
```

Run tests with (assuming you are still under build):
```bash
./test_gradcheck
./test_utils
```

Download and unzip the Wikitext-2 corpus:
```bash
curl https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip --output wikitext-2-v1.zip
unzip wikitext-2-v1.zip
head -n 5 ./wikitext-2/wiki.train.tokens
```

## Training

Learn CBOW embeddings on the training fold with:
```bash
./build/koan -V 2000000 \
             --epochs 10 \
             --dim 300 \
             --negatives 5 \
             --context-size 5 \
             -l 0.075 \
             --threads 16 \
             --cbow true \
             --min-count 2 \
             --file ./wikitext-2/wiki.train.tokens
```

or skipgram embeddings by running with `--cbow false`. `./build/koan` --help for a full list of command-line arguments and descriptions. Learned embeddings will be saved to `embeddings_${CURRENT_TIMESTAMP}.txt` in the present working directory.

After you get your final pre-trained word embedding vectors, you can copy paste to your windows folder. Next, convert it into a word2vec format in order to put it into a gensim model.

Gensim can load two binary formats, word2vec and fastText, and a generic plain text format which can be created by most word embedding tools. The generic plain text format should look like this (in this example 20000 is the size of the vocabulary and 300 is the length of vector).

```console
20000 100
the 0.476841 -0.620207 -0.002157 0.359706 -0.591816 [295 more numbers...]
and 0.223408  0.231993 -0.231131 -0.900311 -0.225111 [295 more numbers..]
[19998 more lines...]
```

Finally, you can do interesting stuff in other NLP tasks, such as this [article]({{site.baseurl}}{% link _posts/2021-01-25-weighted-word-embedding.md %}) I wrote before.

# Conclusion

It is really helpful to make some pre-computed word embedding vectors from scratch, rather than having pre-trained vectors from other websites! Stay Hungry! Stay Foolish! See you next time!