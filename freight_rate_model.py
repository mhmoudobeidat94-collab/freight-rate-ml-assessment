import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


FEATURES = [
    "distance",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "equipment",
    "weight",
    "route_hist_median",
]

TARGET = "posted_rate"

NUMERIC_FEATURES = [
    "distance",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "weight",
    "route_hist_median",
]

CATEGORICAL_FEATURES = ["equipment"]


def make_route_key(df):
    return (
        df["pickup_lat"].astype(str)
        + " -> "
        + df["pickup_lon"].astype(str)
        + " -> "
        + df["delivery_lat"].astype(str)
        + " -> "
        + df["delivery_lon"].astype(str)
    )


def make_model():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                "passthrough",
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    regressor = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )


def add_date_columns(df):
    df = df.copy()

    df["date_dt"] = pd.to_datetime(
        df["date"]
    ).dt.normalize()

    df["month"] = df["date_dt"].dt.month

    df["route_key"] = make_route_key(df)

    return df


def build_strict_temporal_history_features(
    history_df,
    target_df,
    global_median,
):
    """
    Build route historical medians using only observations
    strictly earlier than each target row's date.

    For target rows sharing the same date, the history available
    to all of them is the history from dates strictly before that
    date.

    No same-day or future observations are used.
    """

    history = history_df.copy()
    target = target_df.copy()

    history["date_dt"] = pd.to_datetime(
        history["date"]
    ).dt.normalize()

    target["date_dt"] = pd.to_datetime(
        target["date"]
    ).dt.normalize()

    history["route_key"] = make_route_key(history)
    target["route_key"] = make_route_key(target)

    result = []

    for _, row in target.iterrows():
        route = row["route_key"]
        current_date = row["date_dt"]

        prior = history[
            (history["route_key"] == route)
            & (history["date_dt"] < current_date)
        ][TARGET]

        if len(prior) == 0:
            result.append(global_median)
        else:
            result.append(float(prior.median()))

    target["route_hist_median"] = result

    return target


def train_final_model(dev):
    dev = add_date_columns(dev)

    global_median = float(
        dev[TARGET].median()
    )

    enriched_dev = build_strict_temporal_history_features(
        dev,
        dev,
        global_median,
    )

    # The historical route feature for each training row
    # uses only observations from strictly earlier dates.
    X = enriched_dev[FEATURES]

    y = np.log1p(
        dev[TARGET].astype(float)
    )

    model = make_model()

    model.fit(X, y)

    return model, global_median


def prepare_validation(
    dev,
    validation,
    global_median,
):
    validation = add_date_columns(
        validation
    )

    validation = (
        build_strict_temporal_history_features(
            dev,
            validation,
            global_median,
        )
    )

    return validation


def prepare_december_predictions(
    dev,
    december,
    global_median,
):
    december = december.copy()

    december["date_dt"] = pd.to_datetime(
        december["date"]
    ).dt.normalize()

    # Recover the unique coordinate representation
    # of the requested Lexington -> Fort Wayne route.
    route_rows = dev[
        (dev["pickup"] == "Lexington")
        & (dev["delivery"] == "Fort Wayne")
    ].copy()

    if len(route_rows) == 0:
        raise ValueError(
            "No historical Lexington -> Fort Wayne "
            "route found in training data."
        )

    pickup_coords = route_rows[
        ["pickup_lat", "pickup_lon"]
    ].drop_duplicates()

    delivery_coords = route_rows[
        ["delivery_lat", "delivery_lon"]
    ].drop_duplicates()

    if len(pickup_coords) != 1:
        raise ValueError(
            "Expected exactly one unique "
            "pickup coordinate pair."
        )

    if len(delivery_coords) != 1:
        raise ValueError(
            "Expected exactly one unique "
            "delivery coordinate pair."
        )

    pickup_lat = float(
        pickup_coords.iloc[0]["pickup_lat"]
    )

    pickup_lon = float(
        pickup_coords.iloc[0]["pickup_lon"]
    )

    delivery_lat = float(
        delivery_coords.iloc[0]["delivery_lat"]
    )

    delivery_lon = float(
        delivery_coords.iloc[0]["delivery_lon"]
    )

    december["pickup_lat"] = pickup_lat
    december["pickup_lon"] = pickup_lon

    december["delivery_lat"] = delivery_lat
    december["delivery_lon"] = delivery_lon

    december["route_key"] = make_route_key(
        december
    )

    route_key = make_route_key(
        pd.DataFrame(
            {
                "pickup_lat": [pickup_lat],
                "pickup_lon": [pickup_lon],
                "delivery_lat": [delivery_lat],
                "delivery_lon": [delivery_lon],
            }
        )
    ).iloc[0]

    historical_route = dev[
        dev["route_key"] == route_key
    ].copy()

    # December starts after the development period,
    # so only historical observations before December
    # are allowed to contribute.
    historical_route = historical_route[
        historical_route["date_dt"]
        < december["date_dt"].min()
    ]

    if len(historical_route) == 0:
        december["route_hist_median"] = (
            global_median
        )
    else:
        december["route_hist_median"] = float(
            historical_route[TARGET].median()
        )

    return december


def main(data_dir, output_dir):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = (
        data_dir / "train-test.csv"
    )

    validation_path = (
        data_dir / "validation.csv"
    )

    december_path = (
        data_dir / "december-chart-inputs.csv"
    )

    for path in [
        train_path,
        validation_path,
        december_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

    print("=" * 80)
    print("FREIGHT RATE MODEL")
    print("=" * 80)

    dev = pd.read_csv(train_path)
    validation = pd.read_csv(
        validation_path
    )
    december = pd.read_csv(
        december_path
    )

    print(
        f"Development rows: {len(dev):,}"
    )

    print(
        f"Validation rows:  {len(validation):,}"
    )

    print(
        f"December rows:    {len(december):,}"
    )

    # ------------------------------------------------------------------
    # FINAL MODEL
    # ------------------------------------------------------------------

    model, global_median = (
        train_final_model(dev)
    )

    validation_prepared = prepare_validation(
        dev,
        validation,
        global_median,
    )

    log_predictions = model.predict(
        validation_prepared[FEATURES]
    )

    predictions = np.maximum(
        np.expm1(log_predictions),
        0.0,
    )

    validation_output = pd.DataFrame(
        {
            "load_id": validation[
                "load_id"
            ].values,
            "predicted_rate": predictions,
        }
    )

    validation_output_path = (
        output_dir
        / "validation_predictions.csv"
    )

    validation_output.to_csv(
        validation_output_path,
        index=False,
    )

    print(
        "Validation predictions saved: "
        f"{validation_output_path}"
    )

    # ------------------------------------------------------------------
    # DECEMBER PREDICTIONS
    # ------------------------------------------------------------------

    dev_for_december = add_date_columns(
        dev
    )

    december_prepared = (
        prepare_december_predictions(
            dev_for_december,
            december,
            global_median,
        )
    )

    december_predictions = np.maximum(
        np.expm1(
            model.predict(
                december_prepared[FEATURES]
            )
        ),
        0.0,
    )

    december_output = december.copy()

    december_output[
        "predicted_rate"
    ] = december_predictions

    december_output_path = (
        output_dir
        / "december-chart-inputs.csv"
    )

    december_output.to_csv(
        december_output_path,
        index=False,
    )

    print(
        "December predictions saved: "
        f"{december_output_path}"
    )

    print("\nPrediction summary:")

    print(
        pd.Series(
            predictions
        ).describe()
    )

    print("\nDecember prediction:")

    print(
        pd.Series(
            december_predictions
        ).describe()
    )

    print("\nCOMPLETE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default="data",
        help=(
            "Directory containing "
            "assessment CSV files."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="outputs",
        help=(
            "Directory for generated "
            "predictions."
        ),
    )

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
