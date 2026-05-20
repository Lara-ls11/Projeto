import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

RSS_NO_SIGNAL = 100
RSS_FILL_VALUE = -110.0
RANDOM_STATE = 42


def load_datasets():
    print("A carregar dados...")
    train_x = pd.read_csv("ipin2022_trainrss.csv")
    train_y = pd.read_csv("ipin2022_trainflr.csv").iloc[:, 0]

    val_x = pd.read_csv("ipin2022_validrss.csv")
    val_y = pd.read_csv("ipin2022_validflr.csv").iloc[:, 0]

    test_x = pd.read_csv("ipin2022_testrss.csv")
    test_y = pd.read_csv("ipin2022_testsflr.csv").iloc[:, 0]
    return train_x, train_y, val_x, val_y, test_x, test_y


def align_columns(train_x, val_x, test_x):
    all_columns = sorted(set(train_x.columns) | set(val_x.columns) | set(test_x.columns))
    train_x = train_x.reindex(columns=all_columns, fill_value=np.nan)
    val_x = val_x.reindex(columns=all_columns, fill_value=np.nan)
    test_x = test_x.reindex(columns=all_columns, fill_value=np.nan)
    return train_x, val_x, test_x


def replace_no_signal(df):
    df = df.copy()
    df.replace(RSS_NO_SIGNAL, np.nan, inplace=True)
    return df


def add_rss_features(df):
    df = df.copy()
    values = df.to_numpy(dtype=float)
    valid_mask = ~np.isnan(values)
    strong_thresholds = (-85, -75, -65)

    safe_values = np.where(valid_mask, values, np.nan)
    observed_count = valid_mask.sum(axis=1)
    df["ap_count"] = observed_count
    df["missing_ratio"] = 1.0 - (observed_count / max(df.shape[1], 1))

    df["rss_mean"] = np.nanmean(safe_values, axis=1)
    df["rss_std"] = np.nanstd(safe_values, axis=1)
    df["rss_max"] = np.nanmax(safe_values, axis=1)
    df["rss_min"] = np.nanmin(safe_values, axis=1)
    df["rss_median"] = np.nanmedian(safe_values, axis=1)
    df["rss_range"] = df["rss_max"] - df["rss_min"]

    for q in (10, 25, 75, 90):
        df[f"rss_q{q}"] = np.nanpercentile(safe_values, q, axis=1)

    sorted_desc = np.sort(np.where(valid_mask, values, -999.0), axis=1)[:, ::-1]
    for top_k in (3, 5, 10):
        clipped = sorted_desc[:, :top_k]
        valid_top = clipped > -999.0
        df[f"top{top_k}_mean"] = np.where(
            valid_top.any(axis=1),
            np.nanmean(np.where(valid_top, clipped, np.nan), axis=1),
            RSS_FILL_VALUE,
        )

    for threshold in strong_thresholds:
        df[f"count_gt_{abs(threshold)}"] = np.nansum(safe_values > threshold, axis=1)

    return df.fillna(RSS_FILL_VALUE)


def prepare_data():
    train_x, train_y_raw, val_x, val_y_raw, test_x, test_y_raw = load_datasets()
    train_x, val_x, test_x = align_columns(train_x, val_x, test_x)

    train_x = replace_no_signal(train_x)
    val_x = replace_no_signal(val_x)
    test_x = replace_no_signal(test_x)

    train_x = add_rss_features(train_x)
    val_x = add_rss_features(val_x)
    test_x = add_rss_features(test_x)

    encoder = LabelEncoder()
    train_y = pd.Series(encoder.fit_transform(train_y_raw), index=train_y_raw.index)
    val_y = pd.Series(encoder.transform(val_y_raw), index=val_y_raw.index)

    return train_x, train_y, val_x, val_y, test_x, test_y_raw, encoder
