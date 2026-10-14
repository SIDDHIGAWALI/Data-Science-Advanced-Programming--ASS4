# Employee Segmentation Report

*Method: StandardScaler → PCA (7 components, 87% variance) → K-Means (K=5) validated with Ward hierarchical clustering. Data is synthetic (`src/generate_data.py`), so figures illustrate the method, not a real company.*

## 1. Model selection
- **Elbow / silhouette:** silhouette peaks at **K = 5 (0.383)**; the elbow flattens after 5 (`fig_elbow_silhouette.png`).
- **Hierarchical (Ward):** silhouette 0.373; the dendrogram cut gives 5 clusters (`fig_dendrogram.png`).
- **Agreement:** Adjusted Rand Index between K-Means and Hierarchical = **0.90**, so the segments are stable across algorithms.

## 2. The five segments
| Segment | Headcount | Perf. | Engagement | Overtime h/mo | Absence days | Attrition |
|---|---|---|---|---|---|---|
| High Performers | 235 (19.6%) | 4.5 | 8.1 | 18 | 2.1 | 13% |
| Steady Core | 369 (30.8%) | 3.5 | 7.1 | 8 | 4.0 | 15% |
| New Joiners | 178 (14.8%) | 3.1 | 6.9 | 9 | 3.3 | 16% |
| **Burnout Risk** | 205 (17.1%) | 3.9 | 4.4 | **42** | 9.8 | **83%** |
| **Disengaged Low Performers** | 213 (17.8%) | 2.4 | 3.1 | 6 | 11.2 | **70%** |

Company-wide attrition is 36%.

## 3. High-risk employee clusters
A workforce risk score combines overtime and absenteeism (up) with engagement and manager rating (down). The top 15% (**180 employees**) are flagged: **57% of Burnout Risk** and **30% of Disengaged** employees, and none from the other segments (`fig_risk_matrix.png`).
- *Burnout Risk:* top performers-in-waiting working ~42 h overtime, ~8 projects, long commutes, weak manager ratings.
- *Disengaged Low Performers:* low performance, low training, longest time since promotion.

## 4. Department-level segments
| Department | High-risk share | Attrition | Notable mix |
|---|---|---|---|
| Customer Support | 23% | 45% | 24% Burnout, 25% Disengaged |
| Sales | 18% | 46% | 24% Burnout |
| Engineering | 13% | 33% | 29% High Performers, 20% Burnout |
| Operations | 12% | 29% | 42% Steady Core |
| HR | 11% | 30% | 40% Steady Core |
| Finance | 9% | 26% | 20% Disengaged |

## 5. Performance groups
High Performers are 95% top-rated (≥4). Disengaged Low Performers are 85% rated below 3. Notably **47% of Burnout Risk employees are top-rated**, so the company is at risk of losing strong performers.

## 6. Retention risk groups
Attrition was held out and used only for validation: Burnout Risk (83%) and Disengaged (70%) leave at 4–6× the rate of the other three segments (13–16%), confirming the unsupervised segments capture real retention risk.

## 7. Recommendations
1. **Burnout Risk:** cap overtime, rebalance project load (~8 → 5–6), and coach managers (their manager rating is 3.0 vs 4.4 for High Performers). Prioritise Customer Support and Sales.
2. **Disengaged Low Performers:** structured training and performance-improvement plans, with a clear promotion path (4.1 years since last promotion).
3. **High Performers:** retention bonuses and growth tracks; they are the cheapest group to keep today (13% attrition) and the costliest to lose.
4. **New Joiners:** strengthen onboarding; 37% are rated <3 early, but engagement is healthy and training is high.

## 8. Limitations
Synthetic data with planted structure; cluster sizes and attrition rates will differ on real data. Risk score weights are equal for transparency, not tuned.
