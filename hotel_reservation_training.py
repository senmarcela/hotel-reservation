import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)

# Random seed for my project and test data is 20%
RANDOM_STATE_VALUE = 867
TEST_SIZE = 0.20


# Numeric type features
NUMERIC_FEATURES = [
    "no_of_adults",
    "no_of_children",
    "no_of_weekend_nights",
    "no_of_week_nights",
    "required_car_parking_space",
    "lead_time",
    "arrival_year",
    "arrival_month",
    "arrival_date",
    "repeated_guest",
    "no_of_previous_cancellations",
    "no_of_previous_bookings_not_canceled",
    "avg_price_per_room",
    "no_of_special_requests",
]

# Categorical type features
CATEGORICAL_FEATURES = [
    "type_of_meal_plan",
    "room_type_reserved",
    "market_segment_type",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# Column used to predict (Canceled vs Not_Canceled)
TARGET = "booking_status"


# Read the hotel reservation data into a dataframe
hotel_reservations_data = pd.read_csv("Hotel Reservations.csv")

# Booking_ID is unique for every row so it does not help prediction
hotel_reservations_data = hotel_reservations_data.drop(columns=["Booking_ID"])


# Hotel reservation features
X = hotel_reservations_data[FEATURES]

# Transform booking status target into 0 and 1
booking_status_code = {
    "Not_Canceled": 0,
    "Canceled": 1,
}
y = hotel_reservations_data[TARGET].map(booking_status_code)

# Split input dataset into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=TEST_SIZE, 
                                                    random_state=RANDOM_STATE_VALUE, 
                                                    stratify=y)


print("Total rows in dataset:", len(hotel_reservations_data))
print("Numerical features:", len(NUMERIC_FEATURES))
print("Categorical Features:", len(CATEGORICAL_FEATURES))

print("Missing values in dataset:", int(X.isna().sum().sum()))


print("Target class counts:", hotel_reservations_data[TARGET].value_counts())

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))



# Feature preprocessing - Scaling numeric columns and One-hot encode categorical columns
data_preprocessor = ColumnTransformer(
    transformers=[
        ("numeric_scaling", StandardScaler(), NUMERIC_FEATURES),
        (
            "categorical_one_hot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            CATEGORICAL_FEATURES,
        ),
    ]
)


# Fit preprocessor using training data and Transform train and test data
data_preprocessor.fit(X_train)
X_train_prep = data_preprocessor.transform(X_train)
X_test_prep = data_preprocessor.transform(X_test)
print("Training features shape after preprocessing:", X_train_prep.shape)
print("Testing features shape after preprocessing:", X_test_prep.shape)


# Logistic Regression
logistic_regression_model = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE_VALUE,
)
# train the model
logistic_regression_model.fit(X_train_prep, y_train)
# predict class labels on test data
logistic_regression_test_predictions = logistic_regression_model.predict(X_test_prep)
# predict probabilities for target cancelled class
logistic_regression_test_probabilities = logistic_regression_model.predict_proba(X_test_prep)[:, 1]


# Evaluation metrics
logistic_regression_accuracy = accuracy_score(y_test, logistic_regression_test_predictions)
logistic_regression_auc = roc_auc_score(y_test, logistic_regression_test_probabilities)
logistic_regression_precision = precision_score(y_test, logistic_regression_test_predictions, zero_division=0)
logistic_regression_recall = recall_score(y_test, logistic_regression_test_predictions, zero_division=0)
logistic_regression_f1 = f1_score(y_test, logistic_regression_test_predictions, zero_division=0)
logistic_regression_mcc = matthews_corrcoef(y_test, logistic_regression_test_predictions)
logistic_regression_confusion = confusion_matrix(y_test, logistic_regression_test_predictions)


print("----- Logistic Regression results -----")
print("Accuracy :", round(logistic_regression_accuracy, 4))
print("AUC      :", round(logistic_regression_auc, 4))
print("Precision:", round(logistic_regression_precision, 4))
print("Recall   :", round(logistic_regression_recall, 4))
print("F1       :", round(logistic_regression_f1, 4))
print("MCC      :", round(logistic_regression_mcc, 4))
print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
print(logistic_regression_confusion)


