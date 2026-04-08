# Evolutionary Algorithm for Fraud Rule Optimization
IDS 560 Capstone | UIC MSBA Group 1 | TransUnion

## Project Overview
This project explores whether evolutionary algorithms can automatically generate interpretable, business-ready fraud detection rules from transaction data — replacing slow, manual rule creation with a scalable optimization framework.
Built in partnership with TransUnion, the pipeline implements and compares four algorithmic approaches — Baseline GA, Coevolution GA, NSGA-II, and a Greedy Rule Builder — across 195 model variants on a synthetic fraud dataset of ~330,000 transactions. All models produce human-readable IF-THEN rules directly compatible with production rule engines, evaluated on precision, recall, F1 score, and operational alert rate.
The project was completed as part of IDS 560 at the University of Illinois Chicago, MSBA Program, Spring 2026.

## Team
- Debangana Sanyal
- Adrian Garces
- Sam Chyu
- Anand Mathur
- Siddhi Jain

## Sponsors
Paul Williams, Henry Jonah — TransUnion

## Repository Structure
- `notebooks/` — end-to-end GA pipeline and results visualizations
- `data/` — sample synthetic transaction dataset
- `results/` — all_model_results.csv (195 model variants)
- `docs/` — project documentation and JTA

## How to Run
1. Clone the repository
2. Install dependencies (see below)
3. Open notebooks/Synthetic_Data_Time_GA_Pipeline_IDS560.ipynb
4. Run cells sequentially

## Dependencies
pip install deap scikit-learn pandas numpy matplotlib seaborn

## Key Results

| Model | Precision | Recall | F1 | Alert Rate |
|---|---|---|---|---|
| Baseline GA | 63.4% | 34.2% | 44.4% | 0.27% |
| Coevolution GA | 77.2% | 61.0% | 68.1% | 0.40% |
| NSGA-II (Conservative) | 83.3% | 8.8% | 15.9% | 0.05% |
| NSGA-II (Balanced) | 92.6% | 27.6% | 42.6% | 0.15% |
| NSGA-II (Aggressive) | 39.4% | 69.7% | 50.3% | 0.90% |
| Greedy Builder | 71.7% | 64.5% | 67.9% | 0.46% |


## Notes
- All metrics reported on the test set. Time-based train/val/test split (70/15/15).
- NSGA-II produces a Pareto front — three operating points shown representing different precision/recall trade-offs.
- Full synthetic dataset (~330K transactions) available on request# transunion-fraud-rule-optimization
Evolutionary algorithm pipeline for automated fraud detection rule discovery - UIC MSBA Capstone in partnership with TransUnion.
