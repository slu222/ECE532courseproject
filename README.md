# A Practice of Classification Experiments on Human Activity Recognition Using Smartphones Dataset

1	INTRODUCTION

Machine learning is popular and has been successfully applied in a large amount of fields. Among the various techniques of machine learning, classification is a very basic but useful direction. A lot of algorithms for classification has been created.  In this project,the Human Activity Recognition Using Smartphones Dataset(https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+SmartpHones#) is chosen as the dataset which is conducted experiments on. 


3 kinds of algorithms are picked and implemented to complete the classification task. They are:
•	Least squares method
•	Clustering based method
•	Neural network method


Due to the large size of data, it is necessary to go to the dataset address to download the original data set. This Github page only provides the codes of three algorithms.

2	DATASET

The dataset adopted in this experiment is Human Activity Recognition Using Smartphones Dataset  (Anguita et al., 2013).  The dataset was collected by a series of experiments. The experiments have been carried out with a group of 30 volunteers within an age bracket of 19-48 years. Each person performed six activi- ties (WALKING, WALKING UPSTAIRS, WALKING DOWNSTAIRS, SITTING,STANDING, LAYING) wearing a smartphone (Samsung Galaxy S-II) on the waist. Using its embedded accelerometer and gyroscope, the authors of Anguita et al. (2013) captured 3-axial linear acceleration and 3-axial angular velocity at a con- stant rate of 50Hz. The experiments have been video-recorded to label the data manually. The obtained dataset has been randomly partitioned into two sets, where 70% of the volunteers was selected for generating the training data and 30% the test data.
The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). The sensor acceleration signal, which has gravita- tional and body motion components, was separated using a Butterworth low-pass filter into body acceleration and gravity. The gravitational force is assumed to have only low frequency components, therefore a filter with 0.3 Hz cutoff frequency was used. From each window, a vector of features was obtained by calculating variables from the time and frequency domain.
Finally, there are 10299 samples in the dataset, 7352 for training set, and 2947 for test set. Each sample contains a feature of 561 dimension, and a label ranging from 1 to 6, indicting which activitiy the sample is.

3	LEAST-SQUARES LINEAR REGRESSION ALGORITHM

The least squares method is a linear model, which learns coefficients (or weights) for each feature. It multiplies weights and the values of corresponding feature, adds a bias added, to get a target value. This can be formulated as:

yj = w1*x1 + w2*x2 + . . . + wn*xn + b

where wi and xi is the ith weight and feature value, yj is the target value, n is the number of the feature and b is the bias. The target value yj should be close to the true value. To encourage this, least squared method is proposed. It computes the objective 1/2(y yj)2, and minimizes it. By a series of mathematical operations, we can solve directly the values of the weights and bias. Append b on the first position of the weight vector w, we can solve it by:

w = (XT X)^−1 X^T y

Here,  X  is the matrix whose ith row is the ith sample,  and y is the true label of     all samples. For this and subsequent methods, we employ the Python 1 language to implement our program.


A total of 10299 data were provided in the original data set. Since this project plans to use the Cross-Validation method to divide the data set into four groups, only 10296 data were selected in order to make the amount of the four groups of data consistent. Where the features of all motions are set to the dataset with Suffix X and the labels are set to the dataset with Suffix y. The data is divided into four groups, each of which has 2,574 rows representing 2574 subject motions and 561 columns which are corresponding to the features. 

In those four datasets, three of them were taken as the training dataset and the rest one was saved as the testing dataset. According to the sequence of the data set, the method excluding the first data set is named M1. In the same way, four methods including M1, M2, M3 and M4 are used to select the method with the lowest error rate as the most appropriate method. Because the six actions tested can be roughly divided into dynamic and static. In the operation, the label corresponds to the active state(labels<=3) and the static state(Labels>3) are reassigned to the values of -1 and 1 to train the classifier.

Use the least-squares method with the training dataset to calculate weights for each method. Applying the weights with testing dataset and comparing the predicted labels with the recorded labels to calculate the error rate. The result is shown in Table 1. The corresponding code file name is: least_square.ipynb


Since the lowest value of the result is still close to 50%, similar calculations are carried out for each feature of the data. The results showed that even if only one feature of any 561 was used to calculate the corresponding Weight applied to test dataset, the error rate was still greater than or close to 50%.The reason for this high error rate may be that the features of the data are complex parameters with two attributes of time domain and frequency domain, so the results do not show high similarity in dynamic and static motion states as expected. In other words, due to the influence of factors such as the age and height of the first measured object, sensor parameters in the mobile phone may be unable to display highly consistent representative parameters in static and dynamic states due to the difference in human motion amplitude.

  Methods  	    M1	    M2	    M3	  M4  
  
Error Rate 	 0.770	 0.460	 0.463	0.458

Table 1: Four cross validation results for least squares method.


4 K-MEAN ALGORITHM

The clustering method used is K-Means algorithm (MacQueen et al., 1967; Hartigan & Wong, 1979). Given a set of samples, which can be regarded as points in high dimensional space, K-Means first sample k points as k center points. Then each point will be assigned to the center point which is the closest to itself. By this way,obtaining k clusters  of nodes. For each cluster, getting its center point by computing the mean position of the points in this cluster. Now there are k new center points. Repeating the above procedure until the center points won’t change. It is proved that the iteration can be done in finite steps. Once it is stopped, obtainning k clusters.

The same operation with the first algorithm is used for reading and naming data as for the first algorithm.The difference is that cross-validation Method is not used in this part. Therefore, the selected training data and test data directly use the data provided in the original data set.That is, 70% of the total data is used as a training set and 30% as a test set.

Using K-means on the training set. The k can be set to 2 or 6. The selection of these two values is based on the assumptions that the data will have different centroids
depending on the two static and dynamic states or the more detailed six motion states. 6 represents the original 6 labels; 2 represents the static or active labels. After clustering, each point in the training set has a assignment represent its group index. The points of the same assignment is a class of points. Use the majority of true labels of each class as the label of this class. In the test phase, for each test point, we compute the distance between it and the centers point of each class, and choose the index of the closest center point as the class of this test point. Then we use the label of this class as the predicted label of this test point. The result is shown in Table 2. The corresponding code file name is: kmean(2).ipynb and kmean(6).ipynb

k of 6 represents the problem is a 6-labels classification task. k of2 represents the problem is a binary classification task. The error rate for k of 6 is larger than k of 2,
because more labels increase the complexity of the problem. Compared with the result of least squares method, this clustering based method gain a similar performance. This
may prove that static and dynamic states are not distinct enough when all sensor parameters are taken into account. It is also possible that there are some sensor parameters affected by height, exercise habits, speed of movement or other factors in the different subject states.

   
   K        2       6
   
Error Rate  0.472  0.639

Table 2: Two results for different K values in K-Means algorithm.


5 NEURAL NETWORK ALGORITHM

As mentioned earlier, this part uses the processed training data set and processed test set which has 561 features. Seventy percent of the total data is used to train the network, and the remaining 30 percent is used to calculate the error rate to determine whether the network is working well. This project trained the neural network under two conditions. One is using the original 6 labels for 6 different activities, and another case is reassigning 0 as labels to data with 1-3 labels, and considering these activities as activate state and others are the static state with new label 1.

Using the ReLU function: f(x)=max (0, x). The output of this network is 2 or 6 numbers which denote the probability of 2 or 6 labels (based on the case used for training the system). The loss function used for this model is Cross Entropy Loss which is provided by pytorch. SGD algorithm and Adam algorithm are both tested for 2 labels and 6 labels cases as the back propagation algorithm. Both of those algorithms work well for 2 labels case, given 100% accuracy. Choosing SGD algorithm for 2 label case. The learning rate used is 0.001 and the momentum is 0.9. The code file name is: NeuralNet(2).ipynb


 In 6 labels case, the Adam algorithm shows a better performance than that of the SGD algorithm for different sizes of batch size(Table 3 and Table 4). The batch size is a data extraction method to make gradient descent direction more accurate. It can be implemented by a built-in function from torch.utils. The loss function of Adam algorithm has smaller fluctuation and converges better than the SGD algorithm. What’s more, the error rate of test data set can reach as low as 12.487% when the batch size is 32. So, Adam algorithm and batch size 16 are applied to the model. The code file name is: NeuralNet(6).ipynb
 
This algorithms gives much smaller error rates on both 2 labels and 6 labels cases. This shows that the neural network algorithm may be more suitable for the complex features of the original data.

Batch size	   4	       8	       16	      32	      64	        128

Error rate	41.500%	  16.899%	  12.759%	  12.487%	   40.245%	  53.207%

Table 3: Error rate of Adam algorithm for 6 labels with different batch size.

Batch size	    4	        8	        16	      32	     64	     128

Error rate	 53.207%	 53.377%	 53.241%	 53.920% 	77.796%	  65.796%

Table 4: Error rate of SGD algorithm for 6 labels with different batch size.



