"""XGBoostLSS zero-adjusted Beta (ZIBeta) baseline for the wildfire forecaster.

A tabular counterpart to the GCN->LSTM model in the parent package. Predicts the
same zero-inflated Beta hurdle (gate pi, Beta mean mu, precision phi) h months
ahead, but as a direct multi-horizon gradient-boosted model over hand-engineered
lag / rolling / neighbor / seasonal features instead of a graph-temporal net.

Outputs are written under output/xgb/ to keep them separate from the NN's
output/model/.
"""
