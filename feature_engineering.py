import pandas as pd

def create_features(df):

    df['travel_date'] = pd.to_datetime(df['travel_date'])

    df['day_of_week'] = df['travel_date'].dt.dayofweek
    df['day_of_month'] = df['travel_date'].dt.day
    df['month'] = df['travel_date'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)

    df = df.sort_values(['route_name', 'travel_date'])

    # Only lag_1 (safe for small dataset)
    df['lag_1'] = df.groupby('route_name')['filled_seats'].shift(1)

    # Rolling mean 2 days
    df['rolling_mean_2'] = (
        df.groupby('route_name')['filled_seats']
        .shift(1)
        .rolling(2)
        .mean()
    )

    # Drop rows where lag_1 missing
    df = df.dropna(subset=['lag_1'])

    return df


FEATURE_COLUMNS = [

    'bus_count',
    'total_capacity',
    'day_of_week',
    'day_of_month',
    'month',
    'is_weekend',
    'lag_1',
    'rolling_mean_2'
]
