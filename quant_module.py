import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "Output" / "QuantForecasts"
COLUMN_ALIASES = {
    "date": ("date", "trade_date", "日期", "交易日期"),
    "open": ("open", "开盘"),
    "high": ("high", "最高"),
    "low": ("low", "最低"),
    "close": ("close", "收盘"),
    "volume": ("volume", "成交量"),
}


def _normalize_ohlcv(frame):
    columns = {str(column).strip().lower(): column for column in frame.columns}
    rename = {}

    for standard_name, aliases in COLUMN_ALIASES.items():
        source_name = next(
            (columns[alias.lower()] for alias in aliases if alias.lower() in columns),
            None,
        )
        if source_name is None:
            raise ValueError(
                f"行情数据缺少 {standard_name} 列；支持的列名：{', '.join(aliases)}"
            )
        rename[source_name] = standard_name

    data = frame.rename(columns=rename)[list(COLUMN_ALIASES)].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = (
        data.dropna(subset=list(COLUMN_ALIASES))
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    if data.empty:
        raise ValueError("没有可用的日线数据。")
    if (data["close"] <= 0).any():
        raise ValueError("收盘价中存在小于或等于 0 的数据。")
    return data


def _load_data(symbol, source, csv_path, start_date):
    if source == "csv":
        if not csv_path:
            raise ValueError("CSV 数据源需要指定 CSV 文件路径。")
        path = Path(csv_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"找不到 CSV 文件：{path}")
        return _normalize_ohlcv(pd.read_csv(path))

    try:
        import akshare as ak
    except ImportError as error:
        raise RuntimeError(
            "AkShare 未安装，请运行：python -m pip install "
            "-r requirements-feature-modules.txt"
        ) from error

    history_function = getattr(ak, "stock_zh_a_hist_em", None)
    if history_function is None:
        history_function = getattr(ak, "stock_zh_a_hist", None)
    if history_function is None:
        raise RuntimeError("当前 AkShare 版本不包含 A 股日线历史数据接口。")

    frame = history_function(
        symbol=symbol,
        period="daily",
        start_date=start_date,
        end_date=datetime.now().strftime("%Y%m%d"),
        adjust="qfq",
    )
    if frame is None or frame.empty:
        raise ValueError(
            f"AkShare 没有返回 {symbol} 的数据；请检查代码、网络或数据源状态。"
        )
    return _normalize_ohlcv(frame)


def _build_features(data):
    close = data["close"]
    previous_close = close.shift(1)
    daily_range = (data["high"] - data["low"]).replace(0, np.nan)
    features = pd.DataFrame(index=data.index)

    for period in (1, 2, 5, 10, 20):
        features[f"return_{period}d"] = close.pct_change(period)

    features["gap"] = data["open"] / previous_close - 1
    features["intraday_range"] = (data["high"] - data["low"]) / close
    features["close_position"] = (close - data["low"]) / daily_range
    features["volume_change"] = data["volume"].pct_change()
    features["volume_vs_20d"] = data["volume"] / data["volume"].rolling(20).mean() - 1

    for period in (5, 10, 20):
        features[f"volatility_{period}d"] = close.pct_change().rolling(period).std()

    return features.replace([np.inf, -np.inf], np.nan)


def _make_models():
    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    regressor = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    return classifier, regressor


def _forecast_horizon(data, features, symbol, horizon):
    close = data["close"]
    future_return = close.shift(-horizon) / close - 1
    labeled = features.copy()
    labeled["target_return"] = future_return
    labeled["target_up"] = (future_return > 0).astype(int)
    labeled = labeled.dropna()

    if len(labeled) < 250:
        raise ValueError(
            f"{symbol} 的 {horizon} 日预测仅有 {len(labeled)} 条有效样本；"
            "请提供至少约 1.5 年日线数据，且越长越好。"
        )

    feature_columns = list(features.columns)
    x = labeled[feature_columns]
    y_return = labeled["target_return"]
    y_up = labeled["target_up"]

    split = int(len(labeled) * 0.8)
    train_end = split - horizon
    if train_end < 60 or len(labeled) - split < 10:
        raise ValueError(f"{symbol} 的 {horizon} 日预测数据不足以进行时间序列验证。")

    validation_classifier, validation_regressor = _make_models()
    validation_classifier.fit(x.iloc[:train_end], y_up.iloc[:train_end])
    validation_regressor.fit(x.iloc[:train_end], y_return.iloc[:train_end])

    validation_classes = validation_classifier.predict(x.iloc[split:])
    validation_returns = validation_regressor.predict(x.iloc[split:])
    validation_accuracy = accuracy_score(
        y_up.iloc[split:],
        validation_classes,
    ) * 100
    validation_mae = mean_absolute_error(
        y_return.iloc[split:],
        validation_returns,
    ) * 100

    latest_features = features[feature_columns].dropna()
    if latest_features.empty:
        raise ValueError("历史数据不足，无法生成当前预测。")

    latest_index = latest_features.index[-1]
    reference_close = float(close.iloc[latest_index])
    current_x = latest_features.iloc[[-1]]
    classifier, regressor = _make_models()
    classifier.fit(x, y_up)
    regressor.fit(x, y_return)

    predicted_return = float(regressor.predict(current_x)[0])
    probability_up = 0.0
    if 1 in classifier.classes_:
        up_class_index = list(classifier.classes_).index(1)
        probability_up = float(classifier.predict_proba(current_x)[0][up_class_index])

    return {
        "代码": symbol,
        "数据截至": data.loc[latest_index, "date"].strftime("%Y-%m-%d"),
        "预测周期_交易日": horizon,
        "参考收盘价": reference_close,
        "模型估计收益率_%": predicted_return * 100,
        "模型估计价格": reference_close * (1 + predicted_return),
        "模型估计上涨概率_%": probability_up * 100,
        "验证样本数": len(labeled) - split,
        "验证方向准确率_%": validation_accuracy,
        "验证收益率MAE_%": validation_mae,
    }


def forecast(
    symbol,
    source="akshare",
    csv_path=None,
    horizons=(1,),
    start_date="20180101",
):
    """生成预测数据，不连接交易接口，也不会下单。"""
    if source not in ("akshare", "csv"):
        raise ValueError("source 只能是 akshare 或 csv")
    symbol = str(symbol).strip()
    if not symbol:
        raise ValueError("股票代码不能为空")
    safe_symbol = re.sub(r"[^A-Za-z0-9_-]", "_", symbol).strip("_")
    if not safe_symbol:
        raise ValueError("股票代码只能包含字母、数字、连字符或下划线")
    horizons = sorted(set(int(horizon) for horizon in horizons))
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("预测周期必须是正整数")

    data = _load_data(symbol, source, csv_path, start_date)
    features = _build_features(data)
    results = [
        _forecast_horizon(data, features, symbol, horizon)
        for horizon in horizons
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / (
        f"{safe_symbol}_forecast_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )
    forecasts = pd.DataFrame(results)
    forecasts.to_csv(output_path, index=False, encoding="utf-8-sig")
    return forecasts, output_path
