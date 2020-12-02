# A Practice of Classification Experiments on Human Activity Recognition Using Smartphones Dataset

1	INTRODUCTION

Machine learning is popular and has been successfully applied in a large amount of fields. Among the various techniques of machine learning, classification is a very basic but useful direction. A lot of algorithms for classification has been created.  In this project,the Human Activity Recognition Using Smartphones Dataset(https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+SmartpHones#) is chosen as the dataset which is conducted experiments on. 
3 kinds of algorithms are picked and implemented to complete the classification task. They are:
•	Least squares method
•	Clustering based method
•	Neural network method
#Due to the large size of data, it is necessary to go to the dataset address to download the original data set. This Github page only provides the codes of three algorithms.

2	DATASET

The dataset we adopt in this experiment is Human Activity Recognition Using Smartphones Dataset  (Anguita et al., 2013).  The dataset was collected by a se-  ries of experiments. The experiments have been carried out with a group of 30 volunteers within an age bracket of 19-48 years. Each person performed six activi- ties (WALKING, WALKING UPSTAIRS, WALKING DOWNSTAIRS, SITTING,
STANDING, LAYING) wearing a smartphone (Samsung Galaxy S II) on the waist. Using its embedded accelerometer and gyroscope, the authors of Anguita et al. (2013) captured 3-axial linear acceleration and 3-axial angular velocity at a con- stant rate of 50Hz. The experiments have been video-recorded to label the data manually. The obtained dataset has been randomly partitioned into two sets, where 70% of the volunteers was selected for generating the training data and 30% the test data.
The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). The sensor acceleration signal, which has gravita- tional and body motion components, was separated using a Butterworth low-pass filter into body acceleration and gravity. The gravitational force is assumed to have only low frequency components, therefore a filter with 0.3 Hz cutoff frequency was used. From each window, a vector of features was obtained by calculating variables from the time and frequency domain.
Finally, there are 10299 samples in the dataset, 7352 for training set, and 2947 for test set. Each sample contains a feature of 561 dimension, and a label ranging from 1 to 6, indicting which activitiy the sample is.

3	LEAST  SQUARES METHOD

The least squares method is a linear model, which learns coefficients (or weights) for each feature. It multiplies weights and the values of corresponding feature, adds a bias added, to get a target value. This can be formulated as:
yj = w1x1 + w2x2 + . . . + wnxn + b
where wi and xi is the ith weight and feature value, yj is the target value, n is the number of the feature and b is the bias. The target value yj should be close to the true value. To encourage this, least squared method is proposed. It computes the objective 1/2(y yj)2, and minimizes it. By a series of mathematical operations, we can solve directly the values of the weights and bias. Append b on the first position of the weight vector w, we can solve it by:
w = (XT X)−1XT y
Here,  X  is the matrix whose ith row is the ith sample,  and y is the true label of     all samples. For this and subsequent methods, we employ the Python 1 language to implement our program.
For this method, we practice the cross-validation instead of using the existing training-test split. A total of 10299 data is available.  We  group the 6 labels into  two kinds: static or active. Thus the problem is reduced to a binary classification task. For cross validation, we divide the data set into four groups, only 10296 data were selected in order to make the amount of the four groups of data consistent.    The data is divided into four groups, each of which has a number of 2,574. By permutation and combination, 3/4 data is taken as train dataset and one set of data as the test dataset. Therefore, we can obtain 4 results. Comparing the final error rate, the system corresponding to the lowest value is selected as the final model. The result are presented at Table 1.

Methods	M1	M2	M3	M4 Error Rate	0.493	0.505	0.631	0.493
Table 1: Four cross validation results for least squares method.

The best result is obtained in M1 and M4. However, all 4 results are not that good, which are both close to and even  larger than 50%.   We  claim that the result for   this bad result is the complexity of the data. The features of the data are complex parameters with two attributes of time domain and frequency domain, so the re-  sults do not show high similarity in dynamic and static motion states as expected.In other words, due to the influence of factors such as the age and height of the first measured object, sensor parameters in the mobile phone may be unable to display highly consistent representative parameters in static and dynamic states due to the difference in human motion amplitude.

4	CLUSTERING  BASED METHOD

In this section, we apply a clustering based method to classify the data. The clus- tering method we use is K-Means algorithm (MacQueen et al., 1967; Hartigan & Wong, 1979). Next, we give a brief summary about K-Means, and describe how to use K-Means for classification in detail.
Given a set of samples, which can be regarded as points in high dimensional space, K-Means first sample k points as k center points. Then each point will be assigned to the center point which is the closest to itself. By this way, we obtain k clusters  of nodes. For each cluster, we get its center point by computing the mean position of the points in this cluster. Now we have k new center points. We repeat the above procedure until the center points won’t change. It is proved that the iteration can be done in finite steps. Once it is stopped, we obtain k clusters.
Now we detail how to use K-Means to classify the dataset.  We  run K-means on   the training set. The k can be set to 2 or 6. 6 represents the original 6 labels; 2 represents the static or active labels. After clustering, each point in the training set has a assignment represent its group index.  The points of the same assignment is   a class of points. Use the majority of true labels of each class as the label of this class. In the test phase, for each test point, we compute the distance between it and the centers point of each class, and choose the index of the closest center point as the class of this test point. Then we use the label of this class as the predicted label of this test point.
The result is shown in Table 2. k of 6 represents the problem is a 6-labels classifi- cation task. k of2 represents the problem is a binary classification task. The error rate for k of 6 is larger than k of 2,  because more labels increase the complexity    of the problem. Compared with the result of least squares method, this clustering based method gain a better performance.

k	2	6
Error Rate	0.472	0.655
Table 2: Two result for K-Means algorithm.
