import numpy as np
from ml.trainer import Day1MLTrainer

def extract_crossover_features(stock_data, tick_metrics):
    """
    Extracts quantitative micro-structural features for ML crossover evaluation:
    1. LTQ Ratio (Last Traded Quantity relative to rolling 20m average)
    2. Order Book Depth Imbalance Ratio: (Bid Qty - Ask Qty) / (Bid Qty + Ask Qty)
    3. Volume Surge Ratio: Vol 5m ratio relative to Vol 20m/4
    4. SMMA Spread & Spread Velocity
    5. Price Volatility & Deviation from 20m Average Price
    """
    bid_qty = stock_data.get('total_bid_qty', 0)
    ask_qty = stock_data.get('total_ask_qty', 0)
    total_depth = bid_qty + ask_qty
    
    # Depth Imbalance (-1.0 to +1.0)
    depth_imbalance = (bid_qty - ask_qty) / (total_depth + 1e-5)
    
    smma_20 = tick_metrics.get('smma_20', 0)
    smma_120 = tick_metrics.get('smma_120', 0)
    smma_spread = (smma_20 - smma_120) / (smma_120 + 1e-5)
    
    ltp = stock_data.get('ltp', 0)
    avg_20m = tick_metrics.get('avg_price_20m', ltp)
    price_deviation = (ltp - avg_20m) / (avg_20m + 1e-5)
    
    vol_5m = tick_metrics.get('vol_5m', 0)
    vol_20m = tick_metrics.get('vol_20m', 0)
    expected_5m_vol = (vol_20m / 4.0) + 1e-5
    vol_surge_ratio = vol_5m / expected_5m_vol
    
    # LTQ (Last Traded Quantity) ratio estimate
    ltq_ratio = stock_data.get('ltq_ratio', vol_surge_ratio)

    return {
        'ltq_ratio': float(ltq_ratio),
        'depth_imbalance': float(depth_imbalance),
        'smma_spread': float(smma_spread),
        'price_dev': float(price_deviation),
        'vol_surge_ratio': float(vol_surge_ratio),
        'smma_slope': float(smma_spread * 0.1),
        'total_depth': float(total_depth)
    }

class AICrossoverEvaluator:
    """
    AI/ML Engine to evaluate SMMA(20/120) Crossovers on 1-minute timeframe.
    Uses Day 1 Trained Machine Learning model to predict win probability
    and provides explicit human-readable reasons for trade avoidances.
    """
    def __init__(self):
        self.trainer = Day1MLTrainer()

    def train_model(self):
        return self.trainer.train_on_day1()

    def evaluate_signal(self, signal_type, stock_symbol, stock_data, tick_metrics):
        """
        Evaluates a Buy or Sell 20/120 SMMA crossover signal using Day 1 ML model.
        Returns:
          is_approved: bool
          win_probability: float (0-100%)
          explanation: str detailing why signal is accepted or avoided.
          features: dict of extracted features (including LTQ).
        """
        features = extract_crossover_features(stock_data, tick_metrics)
        reasons = []

        # Predict probability using Day 1 trained model
        ml_prob = self.trainer.predict_probability(features)

        ltq = features['ltq_ratio']
        imbalance = features['depth_imbalance']
        vol_surge = features['vol_surge_ratio']

        # Rule & Permutation checks for explicit rejection explanations
        if signal_type == "BUY":
            if ltq < 1.1:
                reasons.append(f"Avoided: Insufficient LTQ volume expansion ({ltq:.2f}x < 1.1x required threshold).")
            if imbalance < -0.10:
                reasons.append(f"Avoided: Heavy ask wall overhead (Ask depth outweighs Bids by {abs(int(imbalance * 100))}%).")
            if vol_surge < 1.1:
                reasons.append("Avoided: Low 5-min volume acceleration during bullish crossover.")
            if ml_prob < 60.0 and not reasons:
                reasons.append(f"Avoided: Day 1 ML Model predicted low win probability ({ml_prob}% < 60% threshold).")

            if not reasons:
                reasons.append(f"Strong buying volume (LTQ {ltq:.2f}x) & positive bid imbalance (+{int(imbalance * 100)}%).")

        elif signal_type == "SELL":
            if ltq < 1.1:
                reasons.append(f"Avoided: Weak LTQ volume spike on breakdown ({ltq:.2f}x < 1.1x).")
            if imbalance > 0.10:
                reasons.append(f"Avoided: Buyer support resting in order book (+{int(imbalance * 100)}% bid imbalance).")
            if ml_prob < 60.0 and not reasons:
                reasons.append(f"Avoided: Day 1 ML Model predicted low win probability ({ml_prob}% < 60% threshold).")

            if not reasons:
                reasons.append(f"Heavy selling pressure confirmed by LTQ ({ltq:.2f}x) & ask imbalance ({int(imbalance * 100)}%).")

        is_approved = (ml_prob >= 60.0) and (len([r for r in reasons if "Avoided:" in r]) == 0)

        if is_approved:
            summary_reason = f"PROFITABLE TRADE ({ml_prob}% Day 1 ML Confidence): " + " ".join(reasons)
        else:
            summary_reason = f"AVOID TRADE ({ml_prob}% Day 1 ML Probability): " + " ".join(reasons)

        return {
            'signal_type': signal_type,
            'is_approved': is_approved,
            'probability': ml_prob,
            'explanation': summary_reason,
            'features': features
        }

