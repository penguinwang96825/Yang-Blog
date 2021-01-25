---
layout: post
title: TensorFlow 2.0 Installation
subtitle: Step by Step Tutorial for Installing for TF 2.0
cover-img: /assets/img/2020-11-28-installation-for-tf2/jonathan-cooper.jpg
thumbnail-img: /assets/img/2020-11-28-installation-for-tf2/tf.png
readtime: true
show-avatar: false
tags: [Python, TensorFlow, Anaconda, GPU, CUDA]
comments: true
---

TensorFlow makes people love and hate. It is an end-to-end open source platform for Machine Learning and Deep Learning. However, I always have trouble with installing TensorFlow a bunch of times. Thus I decide to share my experience in order to help others to solve this same problem.

## Set up Conda Environment

Conda is a package and environment management tool that allows you to install Python packages on your computer as well as create and manage multiple Python environments, each containing different packages.

1. Install Python from its official [website](https://www.python.org/downloads/release/python-373/).
2. Install [Anaconda](https://www.anaconda.com/distribution/#download-section) 3 for win10.
3. Open command line. (Windows Key +R and type in CMD)
4. Create a virtual environment and change the ENV_NAME of an ipython kernel.

{% highlight bash %}
conda update conda
conda create --name ENV_NAME python=3.7.3
conda activate ENV_NAME 
conda install ipykernel -y
python -m ipykernel install --user --name ENV_NAME --display-name "ENV_NAME"
{% endhighlight %}

## CUDA and cuDNN

If you intend to utilise GPU to speed up your computation, it is neccessary to install CUDA and cuDNN. This [link](https://www.tensorflow.org/install/source#gpu) is the overview of the compatible versions for Tensorflow. Please download the list below first.

- NVIDIA® GPU drivers: [link](https://www.nvidia.com/Download/index.aspx?lang=en-us)
- CUDA® Toolkit: [link](https://developer.nvidia.com/cuda-toolkit-archive)
- cuDNN SDK (Unzip to C:\tools\cuda): [link](https://developer.nvidia.com/cudnn)
- (Optional) TensorRT 5.0: [link](https://developer.nvidia.com/tensorrt)

After installing particular CUDA/cuDNN combination, we have to set environment variable.

![System Properties](https://miro.medium.com/max/618/1*NIVaXFnphn-_xCJr4-snJA.png)

On the Environment Variables dialog, you’ll see two sets of variables: one for user variables and the other for system variables. Just choose the upper one to edit. Next, on the Edit environment variable dialog, you’ll see a list of all the paths that are currently in the PATH variable. Add the following to PATH variable.

{% highlight bash %}
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\extras\CUPTI\libx64
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include
C:\tools\cuda\bin
{% endhighlight %}

There is a shorcut to add PATH variable more efficiently. That is, by adding it via command line. Use Windows Key + R to quickly launch Apps as administrator, and type in CMD. After opening command line, input the following code.

{% highlight bash %}
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin;%PATH%
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\extras\CUPTI\lib64;%PATH%
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include;%PATH%
SET PATH=C:\tools\cuda\bin;%PATH%
{% endhighlight %}

## Final Step
This is the final step. In command line just type in `pip install tensorflow` and it is done. Congratulations!

Hope everyone enjoy Tensorflow!