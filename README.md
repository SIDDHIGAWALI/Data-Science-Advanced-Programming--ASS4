# Employee Workforce Segmentation

Unsupervised segmentation of 1,200 employees using **PCA, K-Means and Hierarchical Clustering**, with risk, department, performance and retention analysis.

## Structure
```
data/            employees.csv (synthetic), employees_pca.csv, employees_segmented.csv
notebooks/       01_pca_report.ipynb, 02_cluster_report.ipynb (executed, outputs included)
reports/         Employee_Segmentation_Report.md, PCA_Summary.md, figures, summary CSVs
dashboard/       workforce_dashboard.html  (open in any browser, self-contained)
src/             generate_data.py, build_notebooks.py, build_dashboard.py
```

## Run it
```bash
pip install -r requirements.txt
python src/generate_data.py
python src/build_notebooks.py
jupyter nbconvert --to notebook --execute --inplace notebooks/01_pca_report.ipynb notebooks/02_cluster_report.ipynb
python src/build_dashboard.py
```

## Key results
- PCA: 12 features → 7 components (87% variance)
- K = 5 (silhouette 0.383); K-Means vs Ward agreement ARI = 0.90
- Segments: High Performers, Steady Core, New Joiners, Burnout Risk, Disengaged Low Performers
- Burnout Risk (83%) and Disengaged (70%) attrition vs 13–16% elsewhere

The dataset is synthetic; swap `data/employees.csv` for real HR data with the same columns to reuse the pipeline.
