# xla


---------------------------------------------------------------------output 1


        Fish       0.00      0.00      0.00         0
        cats       0.00      0.00      0.00         0
        dogs       0.00      0.00      0.00         0
      snakes       1.00      0.18      0.31       724

    accuracy                           0.18       724
   macro avg       0.25      0.05      0.08       724
weighted avg       1.00      0.18      0.31       724


Model: SVM
Training Time: 17.9961 seconds
Accuracy: 18.09%
Precision: 100.00%
Recall: 18.09%
              precision    recall  f1-score   support

        Fish       0.00      0.00      0.00         0
        cats       0.00      0.00      0.00         0
        dogs       0.00      0.00      0.00         0
      snakes       1.00      0.57      0.73       724

    accuracy                           0.57       724
   macro avg       0.25      0.14      0.18       724
weighted avg       1.00      0.57      0.73       724


Model: KNN
Training Time: 0.0136 seconds
Accuracy: 57.32%
Precision: 100.00%
Recall: 57.32%
              precision    recall  f1-score   support

        Fish       0.00      0.00      0.00         0
        cats       0.00      0.00      0.00         0
        dogs       0.00      0.00      0.00         0
      snakes       1.00      0.19      0.31       724

    accuracy                           0.19       724
   macro avg       0.25      0.05      0.08       724
weighted avg       1.00      0.19      0.31       724


Model: Decision Tree
Training Time: 14.8487 seconds
Accuracy: 18.65%
Precision: 100.00%
Recall: 18.65%
Predicted Label by SVM: dogs
Predicted Label by KNN: Fish
Predicted Label by Decision Tree: dogs


-------------------------------------------------------------------------Output2 
              precision    recall  f1-score   support

        Fish       0.00      0.00      0.00         0
        cats       0.00      0.00      0.00         0
        dogs       0.00      0.00      0.00         0
      snakes       1.00      0.18      0.31       724

    accuracy                           0.18       724
   macro avg       0.25      0.05      0.08       724
weighted avg       1.00      0.18      0.31       724


Model: SVM
Training Time: 16.4397 seconds
Accuracy: 18.09%
Precision: 100.00%
Recall: 18.09%
              precision    recall  f1-score   support

        Fish       0.00      0.00      0.00         0
        cats       0.00      0.00      0.00         0
        dogs       0.00      0.00      0.00         0
      snakes       1.00      0.57      0.73       724

    accuracy                           0.57       724
   macro avg       0.25      0.14      0.18       724
weighted avg       1.00      0.57      0.73       724


Model: KNN
Training Time: 0.0025 seconds
Accuracy: 57.32%
Precision: 100.00%
Recall: 57.32%
              precision    recall  f1-score   support

        Fish       0.00      0.00      0.00         0
        cats       0.00      0.00      0.00         0
        dogs       0.00      0.00      0.00         0
      snakes       1.00      0.21      0.34       724

    accuracy                           0.21       724
   macro avg       0.25      0.05      0.09       724
weighted avg       1.00      0.21      0.34       724


Model: Decision Tree
Training Time: 15.6790 seconds
Accuracy: 20.72%
Precision: 100.00%
Recall: 20.72%
Predicted Label by SVM: Fish
Predicted Label by KNN: snakes
Predicted Label by Decision Tree: dogs

--------------------------------------------------------------------------Output3


        Fish       0.00      0.00      0.00         0
        cats       0.00      0.00      0.00         0
        dogs       0.00      0.00      0.00         0
      snakes       1.00      0.18      0.31       724

    accuracy                           0.18       724
   macro avg       0.25      0.05      0.08       724
weighted avg       1.00      0.18      0.31       724


Model: SVM
Training Time: 19.5738 seconds
Accuracy: 18.09%
Precision: 100.00%
Recall: 18.09%
              precision    recall  f1-score   support

        Fish       0.00      0.00      0.00         0
        cats       0.00      0.00      0.00         0
        dogs       0.00      0.00      0.00         0
      snakes       1.00      0.57      0.73       724

    accuracy                           0.57       724
   macro avg       0.25      0.14      0.18       724
weighted avg       1.00      0.57      0.73       724


Model: KNN
Training Time: 0.0140 seconds
Accuracy: 57.32%
Precision: 100.00%
Recall: 57.32%
              precision    recall  f1-score   support

        Fish       0.00      0.00      0.00         0
        cats       0.00      0.00      0.00         0
        dogs       0.00      0.00      0.00         0
      snakes       1.00      0.19      0.32       724

    accuracy                           0.19       724
   macro avg       0.25      0.05      0.08       724
weighted avg       1.00      0.19      0.32       724


Model: Decision Tree
Training Time: 15.6725 seconds
Accuracy: 18.92%
Precision: 100.00%
Recall: 18.92%
Predicted Label by SVM: dogs
Predicted Label by KNN: dogs
Predicted Label by Decision Tree: cats

nhận xét chung 

tỷ lệ nhận dạng sai nhiều < chủ yếu là sai> , nguyên nhân dữ liệu trainning không được nhiều 

em chỉ mới hiểu được thuật toán KNN
svm và Decision Tree  em chưa hiểu được 
 
