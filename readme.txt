编译环境均为Visual Studio
network.cpp network.h为bp神经网络的源代码，编译需要Intel的mkl库
knn.cpp knnm.cpp wknn.cpp为knn算法及优化版的源代码
所有程序均输出到submit.csv

The compilation environment is Visual Studio.
network.cpp and network.h are bp neural network source code, the compiler needs Intel's mkl library.
knn.cpp, knnm.cpp and wknn.cpp are knn algorithm and optimized version of the source code respectively.
All programs output to submit.csv

Introduction
28pixels* 28pixels(784 pixels in total)
42000 images in the training set
28000 images in the testing set
Downloaded from Kaggle(the same as the famous MNIST data)

Improved version of KNN
Binarization
1 (value>=1)
0 (value==0)
finds k nearest neighbors using Euclidean Distance
√ [ Σ( a[i] -b[i] )^2 ] (i= 1，2，…，n)

Improved version of KNN
Improvement
√ [ Σ( a[i] -b[i] )^2 ] (i= 1，2，…，n)
=>Σ( a[i] -b[i] )^2
=> Σ (a[i] xorb[i])
(Because the only value of a[i] and b[i] are 0 and 1)
Use only a bit to store a pixel of the image
For each digit to be recognize, xorwith each digit in the training set, and Distance=the number of 1 in the bitset
#include<bitset>

Improved version of KNN
Performance
K=10
Accuracy：96.2%
Time：119 seconds on i5-5200U
(training set size:42000testing set size:28000)
average:0.00425second per image

Improved version of KNN
value of wjvaries from a maximum of 1 for a nearest neighbor down to a minimum of zero for the most distant of the k neighbors
SahibsinghA. DudaniThe Distance-Weighted k-Nearest-Neighbor Rule
K=12
Accuracy：96.7%

KNNM(k-nearest neighbor mean classifier)
Find k nearest neighbors for each class of training patterns separately, and finds meansfor each of these k neighbors
Classification is done according to the nearest mean pattern
K=5
Accuracy：96.7%

Back Propagation Neural Network
Hidden layer = 1
Hidden node = 300
Mini batch size = 10
Learning Rate =1.5 (*0.8 for every 10 epochs after 30 epochs)
Training set = 42000 = 40000+2000
40000 to train & 2000 to compute the accuracy
End training when accuracy > 96%
Result:97.3%

Increase the number of hidden node can increase accuracy
Initializer of the weight value is important
Perform badly when in [-1,1] uniform distribution(using rand())
Perform better and faster when in (0,0.1) normal distribution
