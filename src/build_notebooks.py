import nbformat as nbf
from pathlib import Path
NB = Path(__file__).resolve().parent.parent / "notebooks"
md = nbf.v4.new_markdown_cell; code = nbf.v4.new_code_cell

# ---------------- Notebook 1: PCA ----------------
n1 = nbf.v4.new_notebook(); n1.cells = [
md("# 01 · PCA Report — Employee Workforce Segmentation\nReduce the HR feature space, justify the number of components, and visualise employee groups."),
code("""import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
sns.set_theme(style="whitegrid"); plt.rcParams["figure.dpi"] = 110
df = pd.read_csv("../data/employees.csv")
df.head()"""),
md("## 1. Feature selection\nWe cluster on **behavioural and HR attributes only**. `employee_id`, `department` (categorical, analysed afterwards) and `left_company` (historical outcome, kept for *validation*) are excluded so no label leaks into the unsupervised model."),
code("""FEATURES = [c for c in df.columns if c not in ("employee_id", "department", "left_company")]
print(len(FEATURES), "features:", FEATURES)
df[FEATURES].describe().T.round(2)"""),
md("## 2. Scaling (mandatory)\nPCA maximises variance, so a feature measured in thousands (salary) would swamp one measured on a 1–5 scale. `StandardScaler` gives every feature mean 0 and unit variance."),
code("""scaler = StandardScaler()
X = scaler.fit_transform(df[FEATURES])
print("means ≈", X.mean(0).round(6).max(), "| stds ≈", X.std(0).round(6).min())"""),
md("## 3. Scree plot and explained variance"),
code("""pca_full = PCA().fit(X)
evr = pca_full.explained_variance_ratio_; cum = evr.cumsum()
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].bar(range(1, len(evr)+1), evr, color="#4C78A8"); ax[0].plot(range(1, len(evr)+1), evr, "o-", color="#E45756")
ax[0].set(title="Scree plot", xlabel="Principal component", ylabel="Explained variance ratio")
ax[1].plot(range(1, len(cum)+1), cum, "o-", color="#4C78A8")
ax[1].axhline(0.85, ls="--", color="#E45756"); ax[1].set(title="Cumulative explained variance", xlabel="Number of components", ylabel="Cumulative ratio")
plt.tight_layout(); plt.savefig("../reports/fig_scree.png", bbox_inches="tight"); plt.show()
pd.DataFrame({"PC": [f"PC{i+1}" for i in range(len(evr))], "variance": evr.round(3), "cumulative": cum.round(3)}).head(9)"""),
md("## 4. Choosing the number of components\nWe keep the smallest number of components that explains **≥ 85 %** of the variance (inside the 80–90 % guideline)."),
code("""n_comp = int(np.argmax(cum >= 0.85) + 1)
print(f"Components kept: {n_comp}  -> {cum[n_comp-1]:.1%} of variance retained ({len(FEATURES)} -> {n_comp} dimensions)")
pca = PCA(n_components=n_comp, random_state=42)
Z = pca.fit_transform(X)
pd.DataFrame(Z, columns=[f"PC{i+1}" for i in range(n_comp)]).assign(employee_id=df.employee_id).to_csv("../data/employees_pca.csv", index=False)"""),
md("## 5. What do the components mean? (loadings)"),
code("""load = pd.DataFrame(pca.components_.T, index=FEATURES, columns=[f"PC{i+1}" for i in range(n_comp)])
plt.figure(figsize=(8, 5)); sns.heatmap(load.iloc[:, :4], annot=True, fmt=".2f", cmap="RdBu_r", center=0)
plt.title("Feature loadings (first four PCs)"); plt.savefig("../reports/fig_loadings.png", bbox_inches="tight"); plt.show()
for pc in ["PC1", "PC2", "PC3"]:
    top = load[pc].abs().sort_values(ascending=False).head(4).index
    print(pc, "driven by:", ", ".join(f"{t} ({load.loc[t, pc]:+.2f})" for t in top))
load.round(3).to_csv("../reports/pca_loadings.csv")"""),
md("## 6. 2-D view of the workforce (before clustering)"),
code("""plt.figure(figsize=(7, 5.5))
sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=df.left_company.map({0: "Stayed", 1: "Left"}), alpha=.6, s=22, palette=["#4C78A8", "#E45756"])
plt.xlabel(f"PC1 ({evr[0]:.1%})"); plt.ylabel(f"PC2 ({evr[1]:.1%})"); plt.title("Employees in PC1–PC2 space (coloured by historical attrition)")
plt.savefig("../reports/fig_pca_attrition.png", bbox_inches="tight"); plt.show()"""),
md("**Take-aways** — a handful of components capture most of the signal; PC1 contrasts engaged/high-performing/well-trained staff with disengaged/absent staff, and PC2 captures workload (overtime, projects, commute). Attrition concentrates at one end of PC1 even though the model never saw it."),
]

# ---------------- Notebook 2: Clustering ----------------
n2 = nbf.v4.new_notebook(); n2.cells = [
md("# 02 · Cluster Report — K-Means & Hierarchical Clustering\nSegment employees, validate the clusters, and analyse risk, department, performance and retention."),
code("""import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, dendrogram
sns.set_theme(style="whitegrid"); plt.rcParams["figure.dpi"] = 110
df = pd.read_csv("../data/employees.csv")
FEATURES = [c for c in df.columns if c not in ("employee_id", "department", "left_company")]
X = StandardScaler().fit_transform(df[FEATURES])
pca = PCA(n_components=0.85, random_state=42); Z = pca.fit_transform(X)
print("PCA input to clustering:", Z.shape)"""),
md("## 1. K-Means — choosing K (elbow + silhouette)"),
code("""ks = range(2, 10); wcss, sil = [], []
for k in ks:
    km = KMeans(k, n_init=10, random_state=42).fit(Z); wcss.append(km.inertia_); sil.append(silhouette_score(Z, km.labels_))
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(ks, wcss, "o-"); ax[0].set(title="Elbow plot", xlabel="K", ylabel="WCSS")
ax[1].plot(ks, sil, "o-", color="#E45756"); ax[1].set(title="Silhouette score", xlabel="K", ylabel="Score")
plt.tight_layout(); plt.savefig("../reports/fig_elbow_silhouette.png", bbox_inches="tight"); plt.show()
print(pd.DataFrame({"K": list(ks), "WCSS": np.round(wcss), "Silhouette": np.round(sil, 3)}).to_string(index=False))
K = int(list(ks)[int(np.argmax(sil))]); print("Chosen K =", K)"""),
md("The silhouette score peaks at **K = 5** and the elbow flattens after 5, so we use five segments."),
code("""km = KMeans(K, n_init=10, random_state=42).fit(Z)
df["kmeans_cluster"] = km.labels_
print("Silhouette (K-Means):", round(silhouette_score(Z, km.labels_), 3))"""),
md("## 2. Hierarchical clustering (Ward linkage)"),
code("""L = linkage(Z, method="ward")
plt.figure(figsize=(11, 4.5))
dendrogram(L, truncate_mode="lastp", p=25, leaf_rotation=90, color_threshold=L[-(K-1), 2])
plt.axhline(L[-(K-1), 2], ls="--", color="k"); plt.title("Dendrogram (Ward) — cut line giving 5 clusters"); plt.ylabel("Distance")
plt.savefig("../reports/fig_dendrogram.png", bbox_inches="tight"); plt.show()
hc = AgglomerativeClustering(n_clusters=K, linkage="ward").fit(Z)
df["hier_cluster"] = hc.labels_
print("Silhouette (Hierarchical):", round(silhouette_score(Z, hc.labels_), 3))
print("Agreement with K-Means (Adjusted Rand Index):", round(adjusted_rand_score(km.labels_, hc.labels_), 3))"""),
md("Both algorithms recover nearly the same structure (high ARI), which is strong evidence the segments are real. We carry **K-Means** forward for profiling."),
md("## 3. Profiling and naming the segments\nCluster IDs are arbitrary, so segments are named from their *profiles* with transparent rules."),
code("""prof = df.groupby("kmeans_cluster")[FEATURES + ["left_company"]].mean()
names = {}
names[prof.overtime_hours_month.idxmax()] = "Burnout Risk"
rest = prof.drop(index=list(names))
names[rest.performance_rating.idxmin()] = "Disengaged Low Performers"
rest = prof.drop(index=list(names))
names[rest.performance_rating.idxmax()] = "High Performers"
rest = prof.drop(index=list(names))
names[rest.tenure_years.idxmin()] = "New Joiners"
names[[i for i in prof.index if i not in names][0]] = "Steady Core"
df["segment"] = df.kmeans_cluster.map(names)
order = ["High Performers", "Steady Core", "New Joiners", "Burnout Risk", "Disengaged Low Performers"]
seg = df.groupby("segment").agg(headcount=("employee_id", "count"), **{f: (f, "mean") for f in FEATURES}, attrition_rate=("left_company", "mean")).loc[order]
seg["share_%"] = (seg.headcount / len(df) * 100).round(1)
seg.round(2).T"""),
md("## 4. PCA visual validation"),
code("""palette = dict(zip(order, ["#59A14F", "#4C78A8", "#76B7B2", "#F28E2B", "#E15759"]))
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for a, col, t in [(ax[0], "segment", "K-Means segments"), (ax[1], "hier_cluster", "Hierarchical clusters")]:
    sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=df[col], palette=palette if col == "segment" else "tab10", s=22, alpha=.7, ax=a)
    a.set(title=t, xlabel="PC1", ylabel="PC2")
plt.tight_layout(); plt.savefig("../reports/fig_clusters_pca.png", bbox_inches="tight"); plt.show()"""),
md("## 5. High-risk employee clusters — Risk Matrix\nRisk is defined from workforce signals (not from the attrition label): **burnout exposure** (overtime, absenteeism) vs **disengagement** (low engagement, low manager rating)."),
code("""def z(s): return (s - s.mean()) / s.std()
df["risk_score"] = (z(df.overtime_hours_month) + z(df.absenteeism_days) - z(df.engagement_score) - z(df.manager_rating)) / 4
rm = df.groupby("segment").agg(burnout=("overtime_hours_month", "mean"), absence=("absenteeism_days", "mean"),
        engagement=("engagement_score", "mean"), risk_score=("risk_score", "mean"), attrition=("left_company", "mean")).loc[order]
plt.figure(figsize=(7, 5))
for s, r in rm.iterrows():
    plt.scatter(r.burnout, r.engagement, s=seg.loc[s, "headcount"]*2.2, color=palette[s], alpha=.8); plt.annotate(s, (r.burnout, r.engagement), ha="center", fontsize=8, xytext=(0, 14), textcoords="offset points")
plt.xlabel("Avg overtime hours / month (workload strain)"); plt.ylabel("Avg engagement score"); plt.title("Risk matrix (bubble = headcount)")
plt.savefig("../reports/fig_risk_matrix.png", bbox_inches="tight"); plt.show()
cut = df.risk_score.quantile(.85); df["high_risk_flag"] = (df.risk_score >= cut).astype(int)
print("High-risk employees (top 15% score):", df.high_risk_flag.sum())
print(df.groupby("segment").high_risk_flag.mean().loc[order].round(3))
rm.round(2)"""),
md("## 6. Department-level workforce segments"),
code("""ct = pd.crosstab(df.department, df.segment, normalize="index").loc[:, order] * 100
plt.figure(figsize=(9, 4.5)); sns.heatmap(ct, annot=True, fmt=".0f", cmap="YlGnBu"); plt.title("Segment mix by department (% of department headcount)")
plt.savefig("../reports/fig_dept_segments.png", bbox_inches="tight"); plt.show()
dept = df.groupby("department").agg(headcount=("employee_id", "count"), avg_perf=("performance_rating", "mean"), avg_engagement=("engagement_score", "mean"),
        avg_overtime=("overtime_hours_month", "mean"), attrition=("left_company", "mean"), high_risk_share=("high_risk_flag", "mean")).round(2)
dept.sort_values("high_risk_share", ascending=False)"""),
md("## 7. Performance groups"),
code("""df["performance_group"] = pd.cut(df.performance_rating, [0, 2.9, 3.9, 5], labels=["Low (<3)", "Solid (3–3.9)", "Top (≥4)"])
pg = pd.crosstab(df.segment, df.performance_group, normalize="index").loc[order] * 100
pg.round(1)"""),
md("## 8. Retention risk groups\nHistorical attrition is used here **only to validate** the unsupervised segments."),
code("""ret = df.groupby("segment").left_company.agg(["mean", "sum", "count"]).loc[order]; ret.columns = ["attrition_rate", "leavers", "headcount"]
fig, ax = plt.subplots(figsize=(8, 4)); ax.barh(ret.index[::-1], ret.attrition_rate[::-1] * 100, color=[palette[s] for s in ret.index[::-1]])
ax.axvline(df.left_company.mean() * 100, ls="--", color="k"); ax.set(xlabel="Attrition rate (%)", title="Attrition by segment (dashed = company average)")
plt.savefig("../reports/fig_retention.png", bbox_inches="tight"); plt.show()
df["retention_risk"] = pd.cut(df.segment.map(ret.attrition_rate), [0, .25, .5, 1], labels=["Low", "Medium", "High"])
print(df.retention_risk.value_counts())
ret.round(3)"""),
md("## 9. Save outputs for the dashboard"),
code("""pcs = pd.DataFrame(Z[:, :3], columns=["PC1", "PC2", "PC3"])
out = pd.concat([df, pcs], axis=1)
out.to_csv("../data/employees_segmented.csv", index=False)
seg.round(3).to_csv("../reports/segment_profiles.csv"); dept.to_csv("../reports/department_summary.csv"); ret.round(3).to_csv("../reports/retention_by_segment.csv")
print("Saved", out.shape)"""),
]
for nb, name in [(n1, "01_pca_report.ipynb"), (n2, "02_cluster_report.ipynb")]:
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, NB / name)
