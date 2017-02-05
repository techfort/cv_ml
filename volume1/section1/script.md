# Introduction to ML and data preparation/feature extraction

Welcome back.
In the previous video we gave an overview of the course, let’s now explore the basics of Machine Learning and its implementation in Computer Vision.

## why machine learning ?

As technology evolves and becomes increasingly prevalent in all aspects of daily life, developers are faced with harder and more challenging tasks that cannot be solved with traditional solutions and algorithms.
Furthermore, we as developers have become more daring, feeling like technological advancements support us in our endeavour to improve the quality of our work and indeed our lives.

I’m sure you found yourself feeling a certain satisfaction when you wrote a program that does something and does it very well, maybe very accurately and very fast.

However, no matter the quality of your program, if you wrote an imperative program that follows a sequence of steps, the accuracy and speed of your program will never improve.

But what if it could? what if you could write programs that performed a certain task and as time passed, the performance of your task improved? Wouldn’t that be amazing?

This is precisely the point of Machine Learning: to write programs that improve the performance of a task over time.

Note, that all too often we are concerned with how fast a certain task can be completed. This is sometimes a measure of performance but not always!
For example, let’s say you wrote a face recognition program, which, given a picture input, outputs the identity of the individual.

You would probably be more concerned with how accurate the face recognition is, than how fast it happened, especially if it was not correct.

So what is Machine Learning in practical terms? 
Machine Learning is a field of computer science that aims at giving computers the ability to improve the performance of a task without being explicitly programmed for it.


What are the main categories of machine learning algorithms?

At the heart of Machine Learning are learning algorithms. These algorithms are divided into several families depending on how they attempt to resolve the learning problem.

Although numerous categories (or families) exist, the three main categories of learning algorithms are:

* supervised learning 
* unsupervised learning
* reinforcement learning

Reinforcement learning is outside the scope of this course, so we will focus our attention on Supervised and Unsupervised Learning.

## Supervised learning

Supervised learning is the family of machine learning algorithms in which a model is trained using training data and expected responses or classifications. 
For example, face recognition is a task that requires supervised learning algorithms, as the program is trained to associate a certain identity to certain pictures, and this association is operated during the training phase.
In summary, supervised learning is a type of learning in which training data is provided with expected results.
Its accuracy is then measured by testing the model on data that was not used for training and checking the correctness of the predictions.
Popular techniques for Supervised learning are Support Vector Machines, Decision Trees, Artificial Neural Networks and more recently Deep Learning Networks.

## Unsupervised learning

Unsupervised learning is the family of machine learning algorithms in which a model practically examines the training data and attempts at grouping (or in machine learning jargon “clustering”) the data in classes.
For example, k-Means, can be used to cluster data in a fixed number (k) of groups, representing classes.
Popular algorithms for unsupervised learning are the aforementioned k-Means and Expectation-Maximization (EM).

### How to choose an algorithm?

The first thing you need to determine is the family of algorithms that are available to you based on the scenario at hand.
For example, if you are not provided with example classification of data, you are most likely dealing with a problem that needs to be resolved with unsupervised learning. From there you can decide whether to use k-Means, EM, or some other algorithm.
On the other end, if you are provided with data and example classification, then both supervised and unsupervised learning might be available to you, with supervised learning being the most likely approach.

As for computer vision, there are numerous elements to be considered before choosing an algorithm: for example, not all approaches are suited to real-time processing. Recognising faces may best be done with SVMs or ANNs while other tasks may be best resolved with DTree.

Also bear in mind that there are really 2 major uses for learning algorithm: prediction and classification.
Regression methods analyse the data and calculate a function that approximately represents the data distribution. So given an arbitrary input you will be able to obtain an output. This is a kind of use of ML that is not very common in computer vision.
Classification methods are most common in computer vision, where the goal is to process an image and classify it as one of many classes available.


