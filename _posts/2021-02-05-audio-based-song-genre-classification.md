---
layout: post
title: Audio-based Song Genre Classification
subtitle: Classify Songs Genres from Audio Data
cover-img: /assets/img/2021-02-05-audio-based-song-genre-classification/yomex-owo.jpg
thumbnail-img: /assets/img/2021-02-05-audio-based-song-genre-classification/wave.png
readtime: true
show-avatar: false
tags: [Python, Librosa, Classification]
comments: true
---

Visualizing sound is kind of a trippy concept. There are some mesmerizing ways to do that, and also more mathematical ones, which I will explore both in this post.

# Introduction

<div class="ready-player-1">
    <audio crossorigin>
        <source src="{{site.baseurl}}{% link /assets/img/2021-02-05-audio-based-song-genre-classification/Beyonce-Halo.mp3 %}" type="audio/mpeg">
    </audio>
</div>

<link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/gh/greghub/green-audio-player/dist/css/green-audio-player.min.css">
<script src="https://cdn.jsdelivr.net/gh/greghub/green-audio-player/dist/js/green-audio-player.min.js"></script>
<script>
    document.addEventListener('DOMContentLoaded', function() {
        new GreenAudioPlayer('.ready-player-1', { showTooltips: true, showDownloadButton: false, enableKeystrokes: true });
    });
</script>


## References

1. https://github.community/t/is-it-possible-to-open-a-sound-file/10377/2
2. https://developer.mozilla.org/en-US/docs/Web/HTML/Element/audio