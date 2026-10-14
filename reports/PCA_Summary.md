# PCA Summary — Employee Workforce Segmentation

**Data:** 1,200 employees, 12 numeric features (age, tenure, salary, performance, engagement, overtime, absenteeism, training, years since promotion, manager rating, projects, commute). `department` and `left_company` were excluded from the model.

**Preprocessing:** `StandardScaler` (mandatory — salary is in thousands, ratings are 1–5).

## Variance explained
| Components | Cumulative variance |
|---|---|
| 1 | 33.7% |
| 2 | 51.8% |
| 3 | 68.3% |
| 5 | 79.9% |
| 6 | 83.8% |
| **7** | **87.1%** (chosen: first count ≥ 85%) |

Dimensionality reduced from **12 → 7** while keeping 87% of the information.

## Interpretation of components
- **PC1 – Engagement & performance axis:** engagement (+0.43), training (+0.40), manager rating (+0.39), performance (+0.35) vs absenteeism (−0.40). Separates thriving from disengaged staff.
- **PC2 – Workload axis:** overtime (+0.61), projects handled (+0.54), performance (+0.32). Isolates over-stretched employees.
- **PC3 – Seniority axis:** tenure (+0.56), age (+0.50), years since promotion (+0.44), salary (+0.33).

## Visual finding
In the PC1–PC2 plane, employees form visible groups, and historical attrition concentrates at the low-PC1 end and the high-PC2 end, even though attrition was never used as an input.

Figures: `fig_scree.png`, `fig_loadings.png`, `fig_pca_attrition.png`. Full code: `notebooks/01_pca_report.ipynb`.
