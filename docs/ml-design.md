# ML design

ML filters strategy candidates only, never orders or position size. Logistic Regression is the baseline; a LightGBM candidate may be added with the optional extra. Models fail closed on missing artifact/schema mismatch. Compare against dummy and rule-only out of sample; threshold selection belongs exclusively to validation data.
