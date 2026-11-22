# Measured results

## Team members

| Team member | Roll number |
|---|---|
| Farhan Naik | 2410110422 |
| Hunar Bhatia | 2410110427 |
| Aniket Sharma | 2410110516 |
| Atul Kumar Arya | 2410110088 |

## Recorded move agreement

| Model | Top 1 | Top 5 | Median time |
|---|---:|---:|---:|
| logistic_regression | 10.6% | 31.5% | 24.9 ms |
| random_forest | 23.1% | 52.0% | 70.4 ms |
| mlp | 26.1% | 53.2% | 20.9 ms |
| cnn | 29.0% | 54.1% | 17.4 ms |
| Combined (equal rank average) | 26.1% | 55.8% | 50.9 ms |

Evaluation: 800 fixed sampled positions from 80 held-out games. Training: 10,240 positions from 640 games. Validation: 1,280 positions from 80 games.
Uniform legal-move baseline: expected top-1 0.42%, top-5 2.11%. This is an analytical reference, not another trained model.

The CNN led on first-choice agreement. The combined method had the highest top-five agreement. The CNN and MLP confidence intervals overlap; this does not establish a statistically significant advantage. Recorded move agreement does not measure perfect play or full-game playing strength.

## Independent engine examples

| Model | Matches engine first choice | Within engine top five |
|---|---:|---:|
| logistic_regression | 1/8 | 2/8 |
| random_forest | 2/8 | 3/8 |
| mlp | 2/8 | 5/8 |
| cnn | 2/8 | 5/8 |

Engine: KataGo v1.16.3 CPU, older b10c128 network, 64 visits per diagram. Chinese rules and 7.5 komi. These eight examples are illustrative only. No historical repetition information is supplied. Engine choices are approximate reference moves, not proven optima.

## Important limitations

- MLP reached the configured maximum of 45 iterations without convergence. The warning is retained in training_log.txt.
- Dataset, model capacity and computation are deliberately limited.
- Candidate models and CNN have different representations and objectives.
- Model scores are ranking values, not calibrated win probabilities.
- No match-based win-rate experiment was performed.
