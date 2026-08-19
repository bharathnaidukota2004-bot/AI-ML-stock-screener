"""
ml/feature_builder.py — Builds the feature vector for the ML model from raw market data.
Centralizes feature extraction so trainer and evaluator use identical features.
"""


def build_features(stock_data: dict, tick_metrics: dict) -> dict:
    """
    Extract all 6 features needed by the CrossoverModel from raw market data.

    Args:
        stock_data: dict with ltp, total_bid_qty, total_ask_qty, ltq_ratio, etc.
        tick_metrics: dict from TickAggregator.get_metrics() with smma_20, vol_5m, etc.

    Returns:
        dict of feature_name -> float
    """
    bid_qty = stock_data.get('total_bid_qty', 0)
    ask_qty = stock_data.get('total_ask_qty', 0)
    total_depth = bid_qty + ask_qty

    # Depth Imbalance: +1 = all bids, -1 = all asks
    depth_imbalance = (bid_qty - ask_qty) / (total_depth + 1e-5)

    smma_20  = tick_metrics.get('smma_20', 0)
    smma_120 = tick_metrics.get('smma_120', 0)
    smma_spread = (smma_20 - smma_120) / (smma_120 + 1e-5)  # normalized spread

    ltp      = stock_data.get('ltp', 0)
    avg_20m  = tick_metrics.get('avg_price_20m', ltp)
    price_dev = (ltp - avg_20m) / (avg_20m + 1e-5)

    vol_5m  = tick_metrics.get('vol_5m', 0)
    vol_20m = tick_metrics.get('vol_20m', 0)
    expected_5m  = (vol_20m / 4.0) + 1e-5
    vol_surge    = vol_5m / expected_5m

    ltq_ratio = stock_data.get('ltq_ratio', vol_surge)

    return {
        'ltq_ratio':       float(ltq_ratio),
        'depth_imbalance': float(depth_imbalance),
        'smma_spread':     float(smma_spread),
        'smma_slope':      float(smma_spread * 0.1),  # approximation from spread velocity
        'vol_surge_ratio': float(vol_surge),
        'price_dev':       float(price_dev),
    }


def features_to_vector(features: dict) -> list:
    """
    Convert feature dict to ordered list for sklearn.
    Order must match CrossoverModel.feature_names!
    """
    return [
        features.get('ltq_ratio', 1.0),
        features.get('depth_imbalance', 0.0),
        features.get('smma_spread', 0.0),
        features.get('smma_slope', 0.0),
        features.get('vol_surge_ratio', 1.0),
        features.get('price_dev', 0.0),
    ]


def build_ghost_signal_data(stock_data: dict, tick_metrics: dict) -> dict:
    """
    Build the dict needed by ghost_detector.detect_ghost().
    Calculates the current spread and approximates average spread from price history.
    """
    bid_price = stock_data.get('best_bid_price', stock_data.get('ltp', 0))
    ask_price = stock_data.get('best_ask_price', stock_data.get('ltp', 0))
    spread = ask_price - bid_price

    # Rough approximation: average spread over 20m is typically 0.5–2x tick size
    # We estimate avg spread as a fraction of the 20m price range
    avg_20m = tick_metrics.get('avg_price_20m', bid_price)
    avg_spread_20m = max(0.05, avg_20m * 0.0003)  # 0.03% of price as baseline

    vol_5m  = tick_metrics.get('vol_5m', 0)
    vol_60m = tick_metrics.get('vol_60m', 1)

    return {
        'spread':         round(spread, 4),
        'avg_spread_20m': round(avg_spread_20m, 4),
        'vol_5m':         float(vol_5m),
        'vol_60m':        float(vol_60m),
    }
