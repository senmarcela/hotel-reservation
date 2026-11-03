# ML Assignment 2 - Hotel Reservations Prediction

## a. Problem statement

Hotels often face losses due to unexpected booking cancellations. The problem is to predict whether a reservation will be canceled or not canceled before the guest's arrival. This prediction helps hotels take preventive actions and improve resource planning.


## b. Dataset description

This project uses the Hotel Reservations dataset, which contains 36,275 booking records with 17 features. The Booking_ID field is removed because it is unique for each booking and does not help in prediction. The target variable is booking_status, which indicates whether a booking was canceled or not canceled.

The dataset contains 14 numerical features and 3 categorical features with no missing values. About 67.2% of the records belong to the Not_Canceled class and 32.8% belongs to Canceled class, making the dataset imbalanced. Since missing a cancellation is usually more costly than a false alarm, class distribution is an important consideration in model evaluation.


## c. Github Repository Link

**GitHub:** [https://github.com/senmarcela/hotel-reservation](https://github.com/senmarcela/hotel-reservation)

Repository contains

```text
Hotel_Reservation/
├── app.py
├── hotel_reservation_training.py
├── requirements.txt
├── README.md
├── Hotel Reservations.csv
├── test_data.csv
└── model/
    ├── data_preprocessor.joblib
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    └── meta.joblib
```

## d. Models used

The models are evaluated using a stratified 20% hold-out test set containing 7,255 records. Model performance is measured using Accuracy, AUC, Precision, Recall, F1-Score, and Matthews Correlation Coefficient (MCC). Precision, Recall, and F1 are calculated with Canceled as the positive class.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.7979 | 0.8571 | 0.7214 | 0.6243 | 0.6694 | 0.5278 |
| Decision Tree | 0.8635 | 0.9291 | 0.8288 | 0.7354 | 0.7793 | 0.6836 |
| KNN | 0.8677 | 0.9193 | 0.8145 | 0.7720 | 0.7927 | 0.6961 |
| Naive Bayes | 0.4057 | 0.7924 | 0.3511 | 0.9600 | 0.5142 | 0.1468 |
| Support Vector Machine | 0.7985 | 0.8574 | 0.7237 | 0.6226 | 0.6694 | 0.5287 |



### Observations on model performance

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Provides performance with 79.79% accuracy. Precision and recall for Canceled are moderate, but it still misses many cancellations false negative = 893 |
| Decision Tree | Achieves highest AUC score of 92.91%. Fewer missed cancellations false negative = 629 |
| KNN | Achieves highest Accuracy 86.77%, F1 score 79.27%, and Matthews Correlation Coefficient score 69.61%. Fewer missed cancellations false negative = 542 |
| Naive Bayes | Recall for Canceled is very high 96%, but it predicts more Canceled false positives 4217. Accuracy and Matthews Correlation Coefficient are very low |
| Support Vector Machine | Provides performance 79.85% accuracy. It also misses many cancellations false negative = 897 |
| Overall, Winner for your dataset? | K-Nearest Neighbor Classifier is the best model for this dataset based on Accuracy, F1, and Matthews Correlation Coefficient |



