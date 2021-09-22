import pandas as pd

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly


def fit_predict(train_df, holidays, prophet_args, horizon=90, period="D", cap=None):
    if cap is not None:
        train_df["cap"] = cap
    model = Prophet(holidays=holidays, **prophet_args)
    model.fit(train_df)

    future = model.make_future_dataframe(horizon, freq=period)
    if cap is not None:
        future["cap"] = cap
    forecast = model.predict(future)

    plot = plot_plotly(model, forecast)
    components = plot_components_plotly(model, forecast)

    forecast.ds = forecast.ds.astype(str)
    train_df.ds = train_df.ds.astype(str)
    forecast = forecast.merge(train_df, on="ds", how="left")
    forecast["ds"] = pd.to_datetime(forecast["ds"])

    # cols = ["ds", "trend", "y", "yhat"]
    # for col in ["yearly", "weekly", "daily", "holidays", "yhat_upper", "yhat_lower"]:
    #     if col in forecast.columns:
    #         cols.append(col)
    return forecast, plot, components
