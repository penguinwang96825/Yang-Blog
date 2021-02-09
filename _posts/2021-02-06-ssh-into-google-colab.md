---
layout: post
title: SSH into Google Colab
subtitle: SSH into Google Colab
cover-img: /assets/img/2021-02-06-ssh-into-google-colab/michael-dziedzic.jpg
thumbnail-img: /assets/img/2021-02-06-ssh-into-google-colab/ssh-keys-cover-image.jpg
readtime: true
show-avatar: false
tags: [SSH, Colab]
comments: true
---

Sometimes the code I write may not work in my local machine, however, it works in Google Colab. So, I wanna connect to Google Colab terminal using SSH.

# Procedure

**Step 1** Create a new notebook in Google Colab.
**Step 2** Copy and paste below code in Colab which installs ngrok and creates a tunnel for us (code taken from this [source](https://gist.github.com/yashkumaratri/204755a85977586cebbb58dc971496da#file-google-colab-ssh)).

```bash
#CODE

#Generate root password
import random, string
password = ''.join(random.choice(string.ascii_letters + string.digits) for i in range(20))

#Download ngrok
! wget -q -c -nc https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
! unzip -qq -n ngrok-stable-linux-amd64.zip
#Setup sshd
! apt-get install -qq -o=Dpkg::Use-Pty=0 openssh-server pwgen > /dev/null
#Set root password
! echo root:$password | chpasswd
! mkdir -p /var/run/sshd
! echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
! echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config
! echo "LD_LIBRARY_PATH=/usr/lib64-nvidia" >> /root/.bashrc
! echo "export LD_LIBRARY_PATH" >> /root/.bashrc

#Run sshd
get_ipython().system_raw('/usr/sbin/sshd -D &')

#Ask token
print("Copy authtoken from https://dashboard.ngrok.com/auth")
import getpass
authtoken = getpass.getpass()

#Create tunnel
get_ipython().system_raw('./ngrok authtoken $authtoken && ./ngrok tcp 22 &')
#Print root password
print("Root password: {}".format(password))
#Get public address
! curl -s http://localhost:4040/api/tunnels | python3 -c \
    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
```

**Step 3** Enter the authorization token which can be found in your ngrok account. So copy and paste it. Then you will be given output as following.

```bash
Copy authtoken from https://dashboard.ngrok.com/auth
··········
Root password: MryWEw9dXuWAHJgKN0O5
tcp://4.tcp.ngrok.io:17523
```

**Step 4** Type into your local machine, and type in the password.

```bash
ssh root@4.tcp.ngrok.io -p 17523
```

![](/assets/img/2021-02-06-ssh-into-google-colab/cmd.jpg)

**Step 5** Run Visual Studio code server. First, mount our Google Drive.

```bash
# Mount Google Drive and make some folders for vscode
from google.colab import drive
drive.mount('/googledrive')
! mkdir -p /googledrive/My\ Drive/colabdrive
! mkdir -p /googledrive/My\ Drive/colabdrive/root/.local/share/code-server
! ln -s /googledrive/My\ Drive/colabdrive /
! ln -s /googledrive/My\ Drive/colabdrive/root/.local/share/code-server /root/.local/share/
```

**Step 6** Install and run the server version of Visual Studio Code.

```bash
! curl -fsSL https://code-server.dev/install.sh | sh > /dev/null
! code-server --bind-addr 127.0.0.1:9999 --auth none &
```

![](/assets/img/2021-02-06-ssh-into-google-colab/vs.png)

## References

1. https://towardsdatascience.com/colab-free-gpu-ssh-visual-studio-code-server-36fe1d3c5243