#!/usr/bin/env python
# coding: utf-8

# In[1]:


Inferential Statistics Ia - Frequentism
Learning objectives
Welcome to the first Frequentist inference mini-project! Over the course of working on this mini-project and the next frequentist mini-project, you'll learn the fundamental concepts associated with frequentist inference. The following list includes the topics you will become familiar with as you work through these two mini-projects:

the z-statistic
the t-statistic
the difference and relationship between the two
the Central Limit Theorem, including its assumptions and consequences
how to estimate the population mean and standard deviation from a sample
the concept of a sampling distribution of a test statistic, particularly for the mean
how to combine these concepts to calculate a confidence interval


# In[2]:


Prerequisites¶
For working through this notebook, you are expected to have a very basic understanding of:

what a random variable is
what a probability density function (pdf) is
what the cumulative density function is
a high-level sense of what the Normal distribution
If these concepts are new to you, please take a few moments to Google these topics in order to get a sense of what they are and how you might use them.

While it's great if you have previous knowledge about sampling distributions, this assignment will introduce the concept and set you up to practice working using sampling distributions. This notebook was designed to bridge the gap between having a basic understanding of probability and random variables and being able to apply these concepts in Python. The second frequentist inference mini-project focuses on a real-world application of this type of inference to give you further practice using these concepts.

For this notebook, we will use data sampled from a known normal distribution. This allows us to compare our results with theoretical expectations.


# In[3]:


I An introduction to sampling from the Normal distribution¶
First, let's explore the ways we can generate the Normal distribution. While there's a fair amount of interest in sklearn within the machine learning community, you're likely to have heard of scipy if you're coming from the sciences. For this assignment, you'll use scipy.stats to complete your work.


# In[4]:


from scipy.stats import norm
from scipy.stats import t
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.random import seed
import matplotlib.pyplot as plt


# In[5]:


get_ipython().set_next_input('Q: Call up the documentation for the norm function imported above. What is the second listed method');get_ipython().run_line_magic('pinfo', 'method')

A: pdf: Probability Density Function Probability density function (PDF) is a statistical expression that defines a probability distribution (the likelihood of an outcome) for a discrete random variable.

Q: Use the method that generates random variates to draw five samples from the standard normal distribution.

A: norm.rvs(size=n) is used to generate Random Variables of size n. norm stands for Normal Distribution.


# In[6]:


seed(47)
# draw five samples here
rvs = norm.rvs(size=5)
rvs


# In[7]:


Q: What is the mean of this sample? Is it exactly equal to the value you expected? Hint: the sample was drawn from the standard normal distribution.
       A: A normal distribution with a mean of 0 and a standard deviation of 1 is called a standard normal distribution. $\bar x$ = 0.19 and $\mu$ is 1 


# In[8]:


# Calculate and print the mean here, hint: use np.mean()
rvsmean = np.mean(rvs)
rvsmean


# In[9]:


Q: What is the standard deviation of these numbers? Calculate this manually here as $\sqrt{\frac{\sum_i(x_i - \bar{x})^2}{n}}$. Hint: np.sqrt() and np.sum() will be useful here and remember that numpy supports broadcasting.

A: The Standard Deviation of the above variables is : 0.96


# In[10]:


sumOfsquares = np.sqrt(np.sum((rvs-rvsmean) ** 2)/5)
sumOfsquares


# In[11]:


Here we have calculated the actual standard deviation of a small (size 5) data set. But in this case, this small data set is actually a sample from our larger (infinite) population. In this case, the population is infinite because we could keep drawing our normal random variates until our computers die. In general, the sample mean we calculate will not be equal to the population mean (as we saw above). A consequence of this is that the sum of squares of the deviations from the population mean will be bigger than the sum of squares of the deviations from the sample mean. In other words, the sum of squares of the deviations from the sample mean is too small to give an unbiased estimate of the population variance. An example of this effect is given here. Scaling our estimate of the variance by the factor $n/(n-1)$ gives an unbiased estimator of the population variance. This factor is known as Bessel's correction. The consequence of this is that the $n$ in the denominator is replaced by $n-1$.

Q: If all we had to go on was our five samples, what would be our best estimate of the population standard deviation? Use Bessel's correction ($n-1$ in the denominator), thus $\sqrt{\frac{\sum_i(x_i - \bar{x})^2}{n-1}}$.

A: As the Sample size is less than 30 and we do not know the Population $\sigma$. The Best estimate is to find t-statistic using Degrees of freedom(df) of sample size(n-1) i.e 4 in this case.


# In[12]:


#  Bessel's Correction
length = len(rvs)-1
stdVal1 = np.sqrt(np.sum((rvs-rvsmean) ** 2)/length)
stdVal1


# In[13]:


get_ipython().set_next_input("Q: Now use numpy's std function to calculate the standard deviation of our random samples. Which of the above standard deviations did it return");get_ipython().run_line_magic('pinfo', 'return')

A: Numpy Standard Deviation Returned the Total size of sample not the Bessel's Correction.


# In[14]:


np.std(rvs)


# In[15]:


Q: Consult the documentation for np.std() to see how to apply the correction for estimating the population parameter and verify this produces the expected result.

A: We can estimate the Population Parameter with the help of t-statistic. i.e $\bar x$ - $\mu$ / $\sigma$/$\sqrt n$

Summary of section
In this section, you've been introduced to the scipy.stats package and used it to draw a small sample from the standard normal distribution. You've calculated the average (the mean) of this sample and seen that this is not exactly equal to the expected population parameter (which we know because we're generating the random variates from a specific, known distribution). You've been introduced to two ways of calculating the standard deviation; one uses $n$ in the denominator and the other uses $n-1$ (Bessel's correction). You've also seen which of these calculations np.std() performs by default and how to get it to generate the other.

You use $n$ as the denominator if you want to calculate the standard deviation of a sequence of numbers. You use $n-1$ if you are using this sequence of numbers to estimate the population parameter. This brings us to some terminology that can be a little confusing.

The population parameter is traditionally written as $\sigma$ and the sample statistic as $s$. Rather unhelpfully, $s$ is also called the sample standard deviation (using $n-1$) whereas the standard deviation of the sample uses $n$. That's right, we have the sample standard deviation and the standard deviation of the sample and they're not the same thing!

The sample standard deviation$$
s = \sqrt{\frac{\sum_i(x_i - \bar{x})^2}{n-1}} \approx \sigma,
$$is our best (unbiased) estimate of the population parameter ($\sigma$).

If your data set is your entire population, you simply want to calculate the population parameter, $\sigma$, via$$
\sigma = \sqrt{\frac{\sum_i(x_i - \bar{x})^2}{n}}
$$as you have complete, full knowledge of your population. In other words, your sample is your population. It's worth noting at this point if your sample is your population then you know absolutely everything about your population, there are no probabilities really to calculate and no inference to be done.

If, however, you have sampled from your population, you only have partial knowledge of the state of your population and the standard deviation of your sample is not an unbiased estimate of the standard deviation of the population, in which case you seek to estimate that population parameter via the sample standard deviation, which uses the $n-1$ denominator.

You're now firmly in frequentist theory territory. Great work so far! Now let's dive deeper.


# In[16]:


II Sampling distributions¶
So far we've been dealing with the concept of taking a sample from a population to infer the population parameters. One statistic we calculated for a sample was the mean. As our samples will be expected to vary from one draw to another, so will our sample statistics. If we were to perform repeat draws of size $n$ and calculate the mean of each, we would expect to obtain a distribution of values. This is the sampling distribution of the mean. The Central Limit Theorem (CLT) tells us that such a distribution will approach a normal distribution as $n$ increases. For the sampling distribution of the mean, the standard deviation of this distribution is given by

$$
\sigma_{mean} = \frac{\sigma}{\sqrt n}
$$
where $\sigma_{mean}$ is the standard deviation of the sampling distribution of the mean and $\sigma$ is the standard deviation of the population (the population parameter).

This is important because typically we are dealing with samples from populations and all we know about the population is what we see in the sample. From this sample, we want to make inferences about the population. We may do this, for example, by looking at the histogram of the values and by calculating the mean and standard deviation (as estimates of the population parameters), and so we are intrinsically interested in how these quantities vary across samples. In other words, now that we've taken one sample of size $n$ and made some claims about the general population, what if we were to take another sample of size $n$? Would we get the same result? Would we make the same claims about the general population? This brings us to a fundamental question: when we make some inference about a population based on our sample, how confident can we be that we've got it 'right'?

Let's give our normal distribution a little flavor. Also, for didactic purposes, the standard normal distribution, with its variance equal to its standard deviation of one, would not be a great illustration of a key point. Let us imagine we live in a town of 50000 people and we know the height of everyone in this town. We will have 50000 numbers that tell us everything about our population. We'll simulate these numbers now and put ourselves in one particular town, called 'town 47', where the population mean height is 172 cm and population standard deviation is 5 cm.


# In[17]:


seed(47)
pop_heights = norm.rvs(172, 5, size=50000)
pop_heights


# In[18]:


_ = plt.hist(pop_heights, bins=30)
_ = plt.xlabel('height (cm)')
_ = plt.ylabel('number of people')
_ = plt.title('Distribution of heights in entire town population')
_ = plt.axvline(172, color='r')
_ = plt.axvline(172+5, color='r', linestyle='--')
_ = plt.axvline(172-5, color='r', linestyle='--')
_ = plt.axvline(172+10, color='r', linestyle='-.')
_ = plt.axvline(172-10, color='r', linestyle='-.')


# In[19]:


get_ipython().set_next_input('Now, 50000 people is rather a lot to chase after with a tape measure. If all you want to know is the average height of the townsfolk, then can you just go out and measure a sample to get a pretty good estimate of the average height');get_ipython().run_line_magic('pinfo', 'height')


# In[ ]:


Now, 50000 people is rather a lot to chase after with a tape measure. If all you want to know is the average height of the townsfolk, then can you just go out and measure a sample to get a pretty good estimate of the average height


# In[20]:


def townsfolk_sampler(n):
    return np.random.choice(pop_heights, n)


# In[21]:


Let's say you go out one day and randomly sample 10 people to measure.


# In[22]:


seed(47)
daily_sample1 = townsfolk_sampler(10)
daily_sample1


# In[23]:


_ = plt.hist(daily_sample1, bins=10)
_ = plt.xlabel('height (cm)')
_ = plt.ylabel('number of people')
_ = plt.title('Distribution of heights in sample size 10')


# In[24]:


get_ipython().set_next_input("The sample distribution doesn't look much like what we know (but wouldn't know in real-life) the population distribution looks like. What do we get for the mean");get_ipython().run_line_magic('pinfo', 'mean')


# In[ ]:


The sample distribution doesn't look much like what we know (but wouldn't know in real-life) the population distribution looks like. What do we get for the mean


# In[25]:


np.mean(daily_sample1)
np.std(daily_sample1)


# In[26]:


get_ipython().set_next_input('And if we went out and repeated this experiment');get_ipython().run_line_magic('pinfo', 'experiment')


# In[ ]:


And if we went out and repeated this experiment


# In[27]:


daily_sample2 = townsfolk_sampler(10)


# In[28]:


np.mean(daily_sample2)


# In[29]:


Q: Simulate performing this random trial every day for a year, calculating the mean of each daily sample of 10, and plot the resultant sampling distribution of the mean.
    


# In[30]:


seed(47)
# take your samples here
RepeatedSampleMean = []

for i in range(365):
    dailySample10 = townsfolk_sampler(10)
    RepeatedSampleMean.append(np.mean(dailySample10))

_ = plt.hist(RepeatedSampleMean, bins=10)
_ = plt.xlabel('height (cm)')
_ = plt.ylabel('number of people')
_ = plt.title('Distribution of heights in sample size 10')
plt.show()


ExpectedMean = np.mean(RepeatedSampleMean)
StandardDeviation = np.std(RepeatedSampleMean)
print ('Mean', ExpectedMean, 'Std', StandardDeviation)


# In[31]:


Mean 171.8660049358649 Std 1.5756704135286475
The above is the distribution of the means of samples of size 10 taken from our population. The Central Limit Theorem tells us the expected mean of this distribution will be equal to the population mean, and standard deviation will be $\sigma / \sqrt n$, which, in this case, should be approximately 1.58.


# In[32]:


Q: Verify the above results from the CLT.

A:


# In[33]:


Climit = 5/np.sqrt(10)
Climit


# In[34]:


Remember, in this instance, we knew our population parameters, that the average height really is 172 cm and the standard deviation is 5 cm, and we see some of our daily estimates of the population mean were as low as around 168 and some as high as 176.

Q: Repeat the above year's worth of samples but for a sample size of 50 (perhaps you had a bigger budget for conducting surveys that year!) Would you expect your distribution of sample means to be wider (more variable) or narrower (more consistent)? Compare your resultant summary statistics to those predicted by the CLT.

A:


# In[35]:


# min(RepeatedSampleMean)
# max(RepeatedSampleMean)
seed(47)

RepeatedSampleMean50 = []

for i in range(365):
    dailySample50 = townsfolk_sampler(50)
    RepeatedSampleMean50.append(np.mean(dailySample50))

RepeatedSampleMean50
ExpectedMean50 = np.mean(RepeatedSampleMean50)
StandardDeviation50 = np.std(RepeatedSampleMean50)
print ('Mean', ExpectedMean50, 'Std', StandardDeviation50)


# In[36]:


Climit = 5/np.sqrt(50)
Climit


# In[37]:


What we've seen so far, then, is that we can estimate population parameters from a sample from the population, and that samples have their own distributions. Furthermore, the larger the sample size, the narrower are those sampling distributions.

III Normally testing times!
get_ipython().set_next_input("All of the above is well and good. We've been sampling from a population we know is normally distributed, we've come to understand when to use $n$ and when to use $n-1$ in the denominator to calculate the spread of a distribution, and we've seen the Central Limit Theorem in action for a sampling distribution. All seems very well behaved in Frequentist land. But, well, why should we really care");get_ipython().run_line_magic('pinfo', 'care')

Remember, we rarely (if ever) actually know our population parameters but you still have to estimate them somehow. If we want to make inferences such as "is this observation unusual?" or "has my population mean changed?" then you need to have some idea of what the underlying distribution is so you can calculate relevant probabilities. In frequentist inference, you use the formulas above to deduce these population parameters. Take a moment in the next part of this assignment to refresh your understanding of how these probabilities work.

Recall some basic properties of the standard Normal distribution, such as about 68% of observations being within plus or minus 1 standard deviation of the mean.

Q: Using this fact, calculate the probability of observing the value 1 or less in a single observation from the standard normal distribution. Hint: you may find it helpful to sketch the standard normal distribution (the familiar bell shape) and mark the number of standard deviations from the mean on the x-axis and shade the regions of the curve that contain certain percentages of the population.

A: Z - Scores +- 1.65 - 90% Confidence Interval +- 1.96 - 95% Confidence Interval +- 2.58 - 99% Confidence Interval

Calculating this probability involved calculating the area under the pdf from the value of 1 and below. To put it another way, we need to integrate the pdf. We could just add together the known areas of chunks (from -Inf to 0 and then 0 to $+\sigma$ in the example above. One way to do this is using look up tables (literally). Fortunately, scipy has this functionality built in with the cdf() function.

Q: Use the cdf() function to answer the question above again and verify you get the same answer.

A:


# In[38]:


def ecdf(sample):
    n = len(sample)
    x = np.sort(sample)
    y = np.arange(1, n+1)/n
    return x, y

x_vers, y_vers = ecdf(RepeatedSampleMean50)
# print (x_vers, y_vers)

# sns.lineplot(x=x_vers, y=y_vers)
_ = plt.plot(x_vers, y_vers, marker='.', linestyle='none')
_ = plt.axvline(172, color='r', linestyle='--')
_ = plt.axhline(0.5, color='r', linestyle='--')
plt.show()


# In[39]:


Q: Using our knowledge of the population parameters for our townsfolk's heights, what is the probability of selecting one person at random and their height being 177 cm or less? Calculate this using both of the approaches given above.

A:


# In[40]:


# type(pop_heights)
# min(RepeatedSampleMean)
max(RepeatedSampleMean)
# NoOfOccur = np.sum(np.asarray(RepeatedSampleMean50) <= 171.94366080916114)
# probVal = NoOfOccur/len(RepeatedSampleMean50)
# probVal


# In[41]:


Q: Turning this question around. Let's say we randomly pick one person and measure their height and find they are 2.00 m tall? How surprised should we be at this result, given what we know about the population distribution? In other words, how likely would it be to obtain a value at least as extreme as this? Express this as a probability.

A:


# In[42]:


We could calculate this probability by virtue of knowing the population parameters. We were then able to use the known properties of the relevant normal distribution to calculate the probability of observing a value at least as extreme as our test value. We have essentially just performed a z-test (albeit without having prespecified a threshold for our "level of surprise")!

We're about to come to a pinch, though here. We've said a couple of times that we rarely, if ever, know the true population parameters; we have to estimate them from our sample and we cannot even begin to estimate the standard deviation from a single observation. This is very true and usually we have sample sizes larger than one. This means we can calculate the mean of the sample as our best estimate of the population mean and the standard deviation as our best estimate of the population standard deviation. In other words, we are now coming to deal with the sampling distributions we mentioned above as we are generally concerned with the properties of the sample means we obtain.

Above, we highlighted one result from the CLT, whereby the sampling distribution (of the mean) becomes narrower and narrower with the square root of the sample size. We remind ourselves that another result from the CLT is that even if the underlying population distribution is not normal, the sampling distribution will tend to become normal with sufficiently large sample size. This is the key driver for us 'requiring' a certain sample size, for example you may frequently see a minimum sample size of 30 stated in many places. In reality this is simply a rule of thumb; if the underlying distribution is approximately normal then your sampling distribution will already be pretty normal, but if the underlying distribution is heavily skewed then you'd want to increase your sample size.

Q: Let's now start from the position of knowing nothing about the heights of people in our town.

Use our favorite random seed of 47, to randomly sample the heights of 50 townsfolk
Estimate the population mean using np.mean
Estimate the population standard deviation using np.std (remember which denominator to use!)
Calculate the (95%) margin of error (use the exact critial z value to 2 decimal places - look this up or use norm.ppf())
Calculate the 95% Confidence Interval of the mean
get_ipython().set_next_input('Does this interval include the true population mean');get_ipython().run_line_magic('pinfo', 'mean')
A:


# In[43]:


seed(47)

sample50 = townsfolk_sampler(50)
samplelen = len(sample50)-1
sampleMean50 = np.mean(sample50)
sampleStd50 = np.sqrt(np.sum((sample50-sampleMean50) ** 2)/samplelen)
print ('Mean - ', sampleMean50, 'Std - ', sampleStd50 )


# In[44]:


#  To Find the 95% Confidence Interval z- value is 1.96
sampleStdError = sampleStd50/np.sqrt(len(sample50))
marginOfError = 1.96*sampleStdError
Intervalfirst = sampleMean50-marginOfError
Intervallast = sampleMean50+marginOfError
print (Intervalfirst, Intervallast)
# We are 95 % Confidence that population Parameter mean is will be in the range of (171.6185985546047 173.9444231607529)


# In[45]:


def ecdf(sample):
    n = len(sample)
    x = np.sort(sample)
    y = np.arange(1, n+1)/n
    return x, y

x_vers, y_vers = ecdf(sample50)
# print (x_vers, y_vers)

_ = plt.plot(x_vers, y_vers, marker='.', linestyle='none')
_ = plt.axvline(172, color='r', linestyle='--')
_ = plt.axhline(0.5, color='r', linestyle='--')
plt.show()


# In[46]:


get_ipython().set_next_input('Q: Above we calculated the confidence interval using the critical z value. What is the problem with this? What requirement, or requirements, are we (strictly) failing');get_ipython().run_line_magic('pinfo', 'failing')

A:

Q: Calculate the 95% confidence interval for the mean using the t distribution. Is this wider or narrower than that based on the normal distribution above? If you're unsure, you may find this resource useful. For calculating the critical value, remember how you could calculate this for the normal distribution using norm.ppf().

A:


# In[47]:


get_ipython().set_next_input('Q: Above we calculated the confidence interval using the critical z value. What is the problem with this? What requirement, or requirements, are we (strictly) failing');get_ipython().run_line_magic('pinfo', 'failing')

A:

Q: Calculate the 95% confidence interval for the mean using the t distribution. Is this wider or narrower than that based on the normal distribution above? If you're unsure, you may find this resource useful. For calculating the critical value, remember how you could calculate this for the normal distribution using norm.ppf().

A:


# In[ ]:




