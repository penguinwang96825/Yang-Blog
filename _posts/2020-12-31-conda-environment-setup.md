---
layout: post
title: Set Up Anaconda for Python
subtitle: The World's Most Popular Data Science Platform
gh-repo: penguinwang96825/Text-Classifier-for-UtaPass-and-KKBOX
cover-img: /assets/img/anaconda.png
tags: [Python, Conda]
comments: true
---

Recently, python is getting more popular, because it can complete a project in a short time. However, setting up virtual environment is crucial for programming several projects. In this article, I will introduce how I setting up a anaconda environment for python.

## Create Conda Environment

When you start learning Python, it is a good starting point to install the newest Python version with the latest versions of the packages you need or want to play around with. Then, most likely, you immerse yourself in this world, and download Python applications from GitHub, Kaggle or other sources. These applications may need other versions of Python/packages than the ones you have been currently using.

1. Install [python](https://www.python.org/downloads/release/python-373/) version 3.7.3

2. Install [Anaconda](https://www.anaconda.com/distribution/#download-section) 3 for win10

3. Create a virtual environment and change the PYTHONPATH of an ipython kernel:
```bash
conda update conda -y
conda create --name my_env python=3.7.3
conda activate MY_ENV
conda install ipykernel -y
python -m ipykernel install --user --name MY_ENV --display-name "MY_ENV"
```

4. GPU support software requirements:
  * NVIDIA® GPU [drivers](https://www.nvidia.com/Download/index.aspx?lang=en-us)
  * [CUDA®](https://developer.nvidia.com/cuda-toolkit-archive) Toolkit
  * [cuDNN](https://developer.nvidia.com/cudnn) SDK
  * (Optional) [TensorRT](https://developer.nvidia.com/tensorrt) 5.0

5. Windows setup
  * Add the CUDA, CUPTI, and cuDNN installation directories to the %PATH% environmental variable. For example, if the CUDA Toolkit is installed to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0 and cuDNN to C:\tools\cuda, update your %PATH% to match:
```bash
$ export PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin;%PATH%
$ export PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\extras\CUPTI\libx64;%PATH%
$ export PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include;%PATH%
$ export PATH=C:\tools\cuda\bin;%PATH%
```
  * Add the absolute path to the TensorRTlib directory to the environment variable LD_LIBRARY_PATH

Enjoy!
