import pandas as pd

def create_features(df):

    df['travel_date'] = pd.to_datetime(df['travel_date'])

    df['day_of_week'] = df['travel_date'].dt.dayofweek
    df['day_of_month'] = df['travel_date'].dt.day
    df['month'] = df['travel_date'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)

    df = df.sort_values(['route_name', 'travel_date'])

    # Lag features for filled seats
    df['lag_1_seats'] = df.groupby('route_name')['filled_seats'].shift(1)
    df['rolling_mean_2_seats'] = (
        df.groupby('route_name')['filled_seats']
        .shift(1)
        .rolling(2)
        .mean()
    )

    # Lag features for price
    if 'average_price' in df.columns:
        df['lag_1_price'] = df.groupby('route_name')['average_price'].shift(1)
        df['rolling_mean_2_price'] = (
            df.groupby('route_name')['average_price']
            .shift(1)
            .rolling(2)
            .mean()
        )

    # Drop rows where lag_1 is missing
    df = df.dropna(subset=['lag_1_seats'])

    return df


FEATURE_COLUMNS_SEATS = [
    'bus_count',
    'total_capacity',
    'day_of_week',
    'day_of_month',
    'month',
    'is_weekend',
    'lag_1_seats',
    'rolling_mean_2_seats'
]

FEATURE_COLUMNS_PRICE = [
    'bus_count',
    'total_capacity',
    'day_of_week',
    'day_of_month',
    'month',
    'is_weekend',
    'lag_1_price',
    'rolling_mean_2_price',
    'lag_1_seats' # Added seats as feature for price
]
