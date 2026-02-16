HYBRID SYSTEMATIC TRADING FRAMEWORK
Machine Learning | Rule-Based Structure | Probabilistic Execution
Executive Summary

This repository contains a hybrid systematic trading framework that integrates:

Quantitative feature engineering

Supervised machine learning models

Structural market rule validation

Probabilistic execution logic

Risk-controlled position management

The framework is designed around one core principle:

Markets are probabilistic environments. Consistency emerges from disciplined edge execution — not prediction certainty.

The system separates signal generation, structural validation, and risk execution into modular components to ensure scalability and research integrity.

Research Philosophy

This project operates under three institutional principles:

Edge over opinion
Signals are statistically derived and validated over large samples.

Execution discipline over outcome attachment
Each trade is one event in a probabilistic distribution.

Risk precedes return
Position sizing and capital preservation are primary system constraints.

The framework avoids discretionary bias and enforces rule-governed execution.

System Architecture
1. Data & Feature Engineering

Historical OHLCV ingestion

Data cleansing & normalization

Volatility, momentum, and structural feature extraction

Optional regime detection inputs

2. Machine Learning Layer

Implemented via ml_strategy.py

Supervised classification/regression modeling

Probability-based outputs (not binary signals)

Confidence scoring thresholds

Model evaluation and validation workflow

Signal example:

probability = model.predict_proba(X)[0][1]

The system interprets probability as expected edge — not guaranteed direction.

3. AI Strategy Pipeline

Implemented via ai_strategy_pipeline.py

Signal filtering

Structural rule alignment

Market condition validation

Execution gating logic

Signals must pass:

Trend bias filter

Volatility constraints

Structural confirmation

Only when probabilistic and structural conditions align does execution proceed.

4. Execution & Risk Control

Fixed fractional risk model

Predefined stop-loss logic

Reward-to-risk validation

Capital exposure constraints

Risk-per-trade governance

The execution layer ensures:

No trade violates capital discipline

Emotional override is structurally restricted

Workflow Structure
notebooks/
    01_data_setup.ipynb
    02_features.ipynb
    03_ml_training.ipynb
    Main.ipynb

src/
    ml_strategy.py
    ai_strategy_pipeline.py
    app.py

Notebooks are used for research and experimentation.
src/ contains structured implementation logic.

Probabilistic Framework

The system assumes:

Any individual trade can lose.

Edge materializes over series.

Statistical expectancy governs performance.

Performance is evaluated across distributions — not isolated outcomes.

Performance Evaluation Metrics

Win Rate

Expectancy

Maximum Drawdown

Risk-Adjusted Return

Distributional Stability

Model Confidence Drift

Deployment Philosophy

The framework is built to evolve toward:

Automated execution

Live broker integration

Adaptive confidence thresholds

Regime-aware signal weighting

Reinforcement learning augmentation

Research Integrity

Modular architecture

Reproducible ML pipeline

Explicit risk controls

Separation of research and execution layers

This design ensures adaptability without compromising system discipline.

Disclaimer

This repository is intended for research and educational purposes only.
Trading financial markets involves substantial risk. Past statistical performance does not guarantee future outcomes.

Author

Kevin Ntabo
Quantitative Systems Developer
Machine Learning | Probabilistic Modeling | Systematic Trading
