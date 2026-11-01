import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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

# Transform booking status target into 0 and 1. 
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(hotel_reservations_data[TARGET])

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





