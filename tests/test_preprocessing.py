import pandas as pd

from src import preprocessing as pp


def make_sample_df():
    return pd.DataFrame([
        {
            'RowNumber': 1,
            'CustomerId': 15634602,
            'Surname': 'Hargrave',
            'CreditScore': 619,
            'Age': 42,
            'Tenure': 2,
            'Balance': 0.00,
            'NumOfProducts': 1,
            'HasCrCard': 1,
            'IsActiveMember': 1,
            'EstimatedSalary': 101348.88,
            'Gender': 'Female',
            'Geography': 'France',
            'Exited': 1,
        },
        {
            'RowNumber': 2,
            'CustomerId': 15647311,
            'Surname': 'Hill',
            'CreditScore': 608,
            'Age': 41,
            'Tenure': 1,
            'Balance': 83807.86,
            'NumOfProducts': 1,
            'HasCrCard': 0,
            'IsActiveMember': 1,
            'EstimatedSalary': 112542.58,
            'Gender': 'Male',
            'Geography': 'Spain',
            'Exited': 0,
        },
        {
            'RowNumber': 3,
            'CustomerId': 15619304,
            'Surname': 'Onio',
            'CreditScore': 502,
            'Age': 42,
            'Tenure': 8,
            'Balance': 159660.80,
            'NumOfProducts': 3,
            'HasCrCard': 1,
            'IsActiveMember': 0,
            'EstimatedSalary': 113931.57,
            'Gender': 'Female',
            'Geography': 'Germany',
            'Exited': 1,
        },
    ])


def test_preprocessing_functions():
    df = make_sample_df()
    df_clean = pp.clean_data(df.copy())
    assert 'RowNumber' not in df_clean.columns
    assert 'CustomerId' not in df_clean.columns
    assert 'Surname' not in df_clean.columns

    x, y = pp.split_features_and_target(df_clean)
    assert 'Exited' not in x.columns
    assert y.shape[0] == x.shape[0]

    cat_cols, num_cols = pp.identify_column_types(x)
    pre = pp.create_preprocessor(cat_cols, num_cols)

    # use small train/test split for transform
    x_train = x.iloc[:2]
    x_test = x.iloc[2:3]
    x_train_t, x_test_t = pp.preprocess_data(x_train, x_test, pre)
    assert x_train_t.shape[0] == x_train.shape[0]
    assert x_test_t.shape[0] == x_test.shape[0]
