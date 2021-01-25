---
layout: post
title: Integrate Atom with Cmder
subtitle: Integrate Atom with Cmder
cover-img: /assets/img/2020-12-15-integrate-atom-with-cmder/vincentiu-solomon.jpg
thumbnail-img: /assets/img/2020-12-15-integrate-atom-with-cmder/atom.png
readtime: true
show-avatar: false
tags: [IDE, Atom, GitHub]
comments: true
---

In this blog, I'll be sharing with you my programming setup for Atom and Cmder, and talking about the most useful tips in it.

# Atom

Atom is a hackable text editor for the 21st century, built on Electron, and based on everything we love about our favorite editors. Compares to VS, the performance differences between tjem come down to a few factors, but one major aspect is the approach with which each app is developed. Visual Studio Code has a tightly controlled core set of functionality, with plugins adding surface-level features. Atom, on the other hand, uses a plugin-based approach to nearly everything. This approach has benefits, but also drawbacks. Atom is slightly slower out of the box, and this only gets worse when adding certain plugins. VS Code has the clear advantage when it comes to performance, but neither editor is slow on a modern machine. This changes when you're editing huge files. Visual Studio Code fares better than Atom, but either is noticeably slow when compared to an editor like Vim or even Sublime Text.

However, I prefer Atom personally. So I will show how to setup Atom in the following blog. Please download [Atom](https://github.com/atom/atom/releases/tag/v1.50.0) first from its GitHub website.

## Install Packages inside Atom

These are the packages I recommended for everyone. They are very powerful and handful for your speed of programming and they helps improve your quality of coding.

1. platformio-ide-terminal (Before installing please pre-install visual studio)
2. atom-beautify
3. tabs-to-spaces
4. git-plus
5. color-picker
6. autocomplete

## Config Setting

In `file` > `Config`, please add the followings,

``` bash
"*":
  core:
    telemetryConsent: "no"
  editor:
    fontSize: 22
    invisibles: {}
    showInvisibles: true
  "exception-reporting":
    userId: "6baad082-b933-4f93-acf5-fea77bb41230"
  "line-ending-selector":
    defaultLineEnding: "LF"
  "platformio-ide-terminal":
    core:
      autoRunCommand: "D:\\cmder\\atom.bat && conda activate nlp && clear"
      shell: "C:\\Windows\\System32\\cmd.exe"
  "tabs-to-spaces":
    onSave: "untabify"
```

# Cmder

Cmder is a software package created out of pure frustration over the absence of nice console emulators on Windows. It is based on amazing software, and spiced up with the Monokai color scheme and a custom prompt layout, looking sexy from the start.

## Integrate with Atom

1. Install platformio-ide-terminal.
2. Create atom.bat in cmder root folder.

```bash
@echo off 
SET CMDER_ROOT=C:\Path\To\cmder
%CMDER_ROOT%\vendor\init.bat
```

3. Settings in platformio-ide-terminal.
 - Auto Run Command: C:\Path\To\cmder\atom.bat && clear
 - Shell Override: C:\Windows\System32\cmd.exe

4. (Optional) If you want to set up conda environment, please go to this [tutorial](https://github.com/penguinwang96825/Set-Up-Conda-Environment) I wrote before. After you build conda environment, set command in Auto Run Command to `C:\Path\To\cmder\atom.bat && conda activate env_name && clear`.

## Conclusion

That's it for todays sharing, if there are more software applications I found useful, I'll post them over here to let you guys know! Cheers, stay tuned!