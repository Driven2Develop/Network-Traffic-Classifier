## This project attempts to provide a Network Classification machine learning algorithm for detection of malicious TCP SYN flood as a distributed denial of service attack.

Intially the dataset is collected from an edge router over a large period of time. Dataset contains about 1.5million samples of network traffic. 
In order to classify the network traffic the follow steps must be completed:
1. The data must be preprocessed to remove any redundant or unnecessary features. As well the samples will be ordered by **ENDTIME**
2. New features are then engineered based on the existing features such as *DURATION* and *UNIQUE PORT COUNT*. The features are generated based on a rolling window.
  * the first type rolling window is based on sample count
  * the second type of rolling window is based on a time frame. 
3. To optimize memory and algorithm performance, the data is discretized so that only whole integers are included in the new dataset. Such procedures include:
  * binning source and destination ports into 10 bins
  * rounding averages to nearest integers
  * changing duration to millisecond count
4. The discretized dataset is split among a 30% training set and a 70% testing set
  * the training set is used to train the model based on the classification labels
  * the testing set is used to test the trained model
5. The Machine learning algorithms are then applied to the split training set, the following algorithms were used
  * Decision tree classification alogorithm
  * Gradient Boost Tree algorithm
  * Support Vector Machines (SVM) algorithm
6. The results are compiled from the three algorithms and evaluated based on the following qualities:
  * precision 
  * accuracy
  * recall

Finally each of the algorithms are compared to determine which algorithm was the best.

### Final Report
The final report contains a more detailed analysis of each of the algorithms and a comparison of their results. 
Improvements are also included in the report to suggest ways to make the algorithm more accurate and precise. A succinct conclusion followed by referenced material is included for suggested readings. 
