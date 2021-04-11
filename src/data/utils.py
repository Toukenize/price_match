from sklearn.preprocessing import LabelEncoder


def encode_label(df, col_to_encode, col_encoded='label_encoded'):

    assert col_to_encode in df.columns, f'{col_to_encode} is not a df column.'

    label_encoder = LabelEncoder()
    df[col_encoded] = label_encoder.fit_transform(df[col_to_encode])

    return df
