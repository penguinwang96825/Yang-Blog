---
layout: post
title: P-Value Easy Explanation
subtitle: Explain the Concept and Calculation of the P-Value through an Example in Hypothesis Testing
cover-img: /assets/img/2021-03-12-easiest-explanation-for-p-value/stephen-dawson.jpg
thumbnail-img: /assets/img/2021-03-12-easiest-explanation-for-p-value/stats.png
readtime: true
show-avatar: false
tags: [Python, Statistics]
comments: true
---

In Data Science interviews, one of the frequently asked questions is **What is P-Value?**. It's hard to grasp the concept behind p-value. To understand p-value, you need to understand some background and context behind it. So, let's start with the basics. When you conduct a piece of quantitative research (such as ML), you are inevitably attempting to answer a research question or hypothesis that you have set. One method of evaluating this research question is via a process called hypothesis testing.

# Introduction

In statistics, the p-value is the probability of obtaining results at least as extreme as the observed results of a statistical hypothesis test, assuming that the null hypothesis is correct. In this article, I will explain the concept and the calculation of the p-value in a simple setting.

# Hypothesis Testing

P-value is used in hypothesis testing process and its value is used in making decision. Let's say we have two opposing statements, one is called null hypothesis (denoted as H0), and the other is called alternative hypothesis (denoted as H1). A null hypothesis is a type of conjecture proposes that there's no difference between certain characteristics of a population (generally about population value being equal to something), and an alternative hypothesis proposed that there's a difference.

Let's take an simple example first. We assume that, we want to see if the machine learning average score of CSSLP students is equal to 60, that is, the population mean $\mu$ is equal to 60. So perhaps the CSSLP students made a great effort in studying and it is believed that the H0 is $\mu=60$. As for the alternative hypothesis believes that the mean has changed and the score is now more than 60, that is, we have H1 of $\mu>60$.

$$
H0: \mu=60
$$

$$
H1: \mu>60
$$

The idea here is that we will take a random sample from this population (all CSSLP students), examine the sample and decide whether the samples support our null hypothesis or the alternative hypothesis. So the logic of hypothesis testing indicates that if null hypothesis is true, then the sample mean (suppose you take a random sample) and calculate the sample mean under the null hypothesis, the sample mean should be close to 60, because sample mean and population mean are expected to be close. However, if sample mean is significantly higher than 60, then we will reject null hypothesis and establish the alternative.

Let's suppose that we took a random sample of 36 CSSLP students from population and calculate the sample mean, and let's suppose the sample mean is 62 ($\overline{X}=62$) and suppose that we know the standard deviation is 4 ($\sigma=4$). If you recall, the standard deviation is a measure of variability, not all students having the same score, so this $\sigma$ here is the 'population' standard deviation in this problem.

Now the question is our $\overline{X}=62$ significantly high? How do we decide if it is significantly high? In statistics, we calculate the probability or the likelihood of the occurence of an outcome. So what we would like to calculate is that if the null hypothesis is true (if the population mean is actually 60), what is the likelihood that $\overline{X}$ would be as high or higher than 62? Let's translate it into a mathematical formula.

<div>
    <span>
        \begin{align}
            P(\overline{X} \geq 62 | \mu = 60) &= P(Z \geq \frac{62-60}{\frac{4}{\sqrt{36}}}) \\
								   			   &= P(Z \geq 3)
        \end{align}
    </span>
</div>

Since $\overline{X}$ has a normal distribution, because $n$ is large we can change this value 60 to a z-score, which is measured in terms of standard deviations from the mean developed by Professor Edward Altman at New York University. It turns out we are calculating the probability that $Z$ is greater than or equal to 3. And how do we calculate this? One way is to read the normal table or we could use the graphing calculator.

Below is the typical graph of z-scores.

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

plt.figure(figsize=(15, 6))
plt.plot(np.linspace(-4, 4, 100), st.norm.pdf(np.linspace(-4, 4, 100)))
plt.grid()
plt.show()
```

![](/assets/img/2021-03-12-easiest-explanation-for-p-value/z-distribution.png)

We want to calculate the area under the standard normal curve to the right of 3. This can denoted with the equation below.

$$
\int_{z}^{\infty} \frac{1}{\sqrt{2 \pi}} e^{- \frac{x^2}{2}} \,dx
$$

```python
from scipy.integrate import quad

def normal_distribution_pdf(x):
    constant = 1.0 / np.sqrt(2*np.pi)
    return(constant * np.exp((-x**2)/2.0))

percentile, _ = quad(normal_distribution_pdf, 3.0, np.Inf)
```

We can now get the percentile of 0.13%. This probability is called the `p-value`. To sum up, P-value is the probability of observing a result as high or higher than what we have observed if the null hypothesis is true. We can then conclude that, under the null hypothesis, what we have observed is highly unlikely to happen. That means null hypothesis and what we have observed, they don't match. Because what we have observed is very unlikely to happen under null hypothesis, we reject null hypothesis and conclude that the alternative is true, which is in this case, $\mu$ is bigger than 60.

However, another question pops out, how do we know the p-value is low enough? Do we have a guideline or threshold to determine this? And the answer is "YES"! Generally, if the probability of an outcome is lower than 5% (so called level of significance or alpha level) that we select in advance, we then consider that probability to be low. In our case, we calculate the p-value as 0.13%, which is obviously way lower than 5% (null hypothesis is unlikely to happen), so we reject the null hypothesis (we say the result is statistically significant).

So far, this is the concept of p-value. Let me mention one thing which is also equally important. Come back to this $\sigma$ value, which is the population standard deviation. Usually, population standard deviation is not given. That means, when we take a sample of 36 observations from the population, we not only calculate the sample mean $\overline{X}$, but we also have to compute the sample standard deviation $S$. If this is the case, then some of the steps will change. Let's compute $S$ from the sample and let's suppose $S$ was also equal to 4 ($S$ and $\sigma$ don't have to be equal).

The problem still remains the same, to calculate the probability of $\overline{X} \geq 62$ given $\mu$ is equal to 60. But the result is no longer called z-score, instead, it's called a t-score. Now the question will be what is the probability that a t-score is bigger than 3.

<div>
    <span>
        \begin{align}
            P(\overline{X} \geq 62 | \mu = 60) &= P(t \geq \frac{62-60}{\frac{4}{\sqrt{36}}}) \\
								   			   &= P(t \geq 3)
        \end{align}
    </span>
</div>

Right now, t distribution graph is very similar to z distribution graph, but the actual shape depends on the sample size. I will show you how a t distribution graph look like in the below figure.

```python
from scipy.stats import t

plt.figure(figsize=(15, 6))
for df in range(10):
    plt.plot(np.linspace(-4, 4, 100), 
    		 t.pdf(np.linspace(-4, 4, 100), df+1), 
    		 label=f"degree={df+1}")
plt.legend(loc="upper right")
plt.grid()
plt.show()
```

![](/assets/img/2021-03-12-easiest-explanation-for-p-value/t-distribution.png)

We got as a refresher that we will read the t-value, or the area above 3 with 35 degrees of freedom ($36-1=35$).

The probability density function for t distribution is:

$$
f(x, n) = \frac{\Gamma (\frac{n+1}{2})}{\sqrt{\pi n} \Gamma (\frac{n}{2})} (1 + \frac{x^2}{n})^{- \frac{n+1}{2}}
$$

where $x$ is a real number and the degrees of freedom parameter $n$ satisfies $n > 0$. $\Gamma$ is the gamma function.

Let's calculate the area under t distribution to the right of 3.

```python
percentile, _ = quad(t.pdf, 3.0, np.Inf, args=(35))
```

This area turns out to be 0.2474%, and this probability is lower than alpha 5%. So our decision will still be the same, what we have observed under null hypothesis is highly unlikely to happen, this means that our assumption that the null hypothesis is correct is most likely to be false, so the null hypothesis should be rejected.

# Conclusion

Hypothesis testing is important not just in data science, but in every field. In this post, we know how to calculate the p-value by hand and also by using Python. Happy learning! Cheers!

## References

1. https://www.analyticsvidhya.com/blog/2020/07/hypothesis-testing-68351/
2. https://www.machinelearningplus.com/statistics/what-is-p-value/
3. https://www.youtube.com/watch?v=kx0xLnqJ_30&list=PLCZeVeoafktVGu9rvM9PHrAdrsUURtLTo&index=1&t=9s&ab_channel=ChandChauhan