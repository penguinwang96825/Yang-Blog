---
layout: post
title: GitHub Pages from Zero to Hero
subtitle: Weighted Word Embedding for Downstream NLP Tasks
cover-img: /assets/img/2021-01-25-github-page-from-zero-to-hero/path.jpg
share-img: /assets/img/2021-01-25-github-page-from-zero-to-hero/path.jpg
thumbnail-img: /assets/img/2021-01-25-github-page-from-zero-to-hero/page.png
readtime: true
show-avatar: false
tags: [GitHub, Jekyll, Cloudflare, Freenom]
---

This article aimes to build website for your project, hosted directly from your GitHub repository. GitHub offers a web hosting service whereby you can serve a static website from a GitHub repository. This platform, GitHub Pages, can be used with CloudFlare whilst using a custom domain name. In this tutorial, I will demonstrate how to use GitHub, Freenom, and Cloudflare together. By taking advantage of CloudFlare’s global network, you can utilise our CDN service to improve your site's performance and security.

# Jekyll

First, select your favorite [Jekyll](https://jekyllthemes.io/free) themes template. There are several free themes options, you can pick whatever you want. Some of them charges a bit, it really depends on you!

Second, fork the project you pick by clicking the `Fork` button at the top right corner of the page. Forking means that you now copied this entired project and all the files into your repository.

Third, name the preoject to `<yourusername>.github.io`. Click on `Settings` at the top and on that page you'll have option to rename the project. After doing so, you will automatically set up your GitHub Pages.

## Configuration

1. Pull your repository to your local folder.

{% highlight bash linenos %}
git clone git@github.com:<github_account_name>/<repo_name>.git
cd <repo_name>
{% endhighlight %}

2. Customize your website settings. Edit the `_config.yml` file to change any settings you want.

3. If you want to run your website locally, you should follow instructions from [here](https://beautifuljekyll.com/getstarted/#install-steps-hard). This is hard to set up initially, but once you complete all instuctions, it makes it super easy to keep your site up to date and you can debug your code locally by using `bundle exec jekyll serve` command.

## Start blogging!

To add pages to your site, you can either write a markdown file (`.md`) or you can write an HTML file (`.html`). Files you create inside the `_posts` directory will be treated as blog entries. Note the format of the blog post files - they must follow the naming convention of `YEAR-MONTH-DAY-title.md`. After you successfully add your own post, you can delete the existing files inside `_posts` to remove the sample posts.

# Get Free Domain

## Freenom

Freenom is the world's first and only free domain provider. You can set up or update certain DNS records and your repository settings to point the default domain for your GitHub Pages site to a custom domain.

1. Go to https://www.freenom.com
2. Check `Availability`.
![](/assets/img/2021-01-25-github-page-from-zero-to-hero/availability.png)
3. If the domain name is available click `Get it now!` and the `Checkout`.
![](/assets/img/2021-01-25-github-page-from-zero-to-hero/freedomains.png)
4. Set the period to 12 months the click `Continue`.
![](/assets/img/2021-01-25-github-page-from-zero-to-hero/checkout.png)
5. Check `I have read and agree to the Terms & Conditions`. Then click `Complete Order`.
![](/assets/img/2021-01-25-github-page-from-zero-to-hero/complete.png)


## Cloudflare

Cloudflare is one of the biggest networks operating on the Internet. People use Cloudflare services for the purposes of increasing the security and performance of their web sites and services.

1. Go to https://www.cloudflare.com and create an account.
2. Click `Add site` in `Account Dashboard` and put your site domain.
![](/assets/img/2021-01-25-github-page-from-zero-to-hero/addsite.png)
3. Select `Free` plan and click `Confirm plan`.
4. Wait till the Cloudflare checking the exsiting DNS record and click `Continue`.
5. Cloudflare will give you two nameservers that you need to set in Freenom.
![](/assets/img/2021-01-25-github-page-from-zero-to-hero/cloudflare.png)
6. Go back to Freenom `Click Service > My Domains`.
![](/assets/img/2021-01-25-github-page-from-zero-to-hero/domain.png)
7. Click `Management Tools > Nameservers`.
![](/assets/img/2021-01-25-github-page-from-zero-to-hero/nameserve.png)
8. Select `Use custom nameservers` and add the cloudflare nameservers, which you get from step 5, to custom name servers.
9. Finish Setting up your Domain on Cloudflare and go to the Domain Dashboard. Open the `Cloudflare Settings` for your domain, and change the SSL Setting to `Flexible SSL`.
![](/assets/img/2021-01-25-github-page-from-zero-to-hero/ssl.png)

## GitHub Pages

Go to github and select the repository you want to add custom domain and select settings. Scroll to bottom you will see section `Custom Domain`. Add your domain and save, now all are set and it will take a while, so please be patient.
![](/assets/img/2021-01-25-github-page-from-zero-to-hero/success.png)

## Final Step

### Letting Search Engines know

In your `_config.yml`, add these:

{% highlight bash linenos %}
url: https://www.yoursite.com   # with the https protocol
enforce_ssl: www.yoursite.com   # without any protocol

# For example, my configuration is this:
url: https://nlper.ml
enforce_ssl: nlper.ml
{% endhighlight %}

Make sure you have a canonical link in your `<head>`:

{% highlight javascript linenos %}
<link rel="canonical" href=" { { site.url } }{ { page.url } }" />
{% endhighlight %}

### Redirect users to https

Add this script to the very top of your `<head>`:

{% highlight javascript linenos %}
<script type="text/javascript">
    var host = "yoursite.com";
    if ((host == window.location.host) && (window.location.protocol != "https:"))
        window.location.protocol = "https";
</script> 
{% endhighlight %}

# Conclusion
After Done You can enter your Domain name on browser and check. Hope this article help you build your website! Enjoy your Free Domain ❤