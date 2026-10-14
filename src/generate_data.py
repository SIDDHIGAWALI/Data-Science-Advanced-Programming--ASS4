"""Generate a synthetic HR dataset (1,200 employees) with latent workforce segments."""
import numpy as np, pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
N = 1200
DEPTS = ["Engineering", "Sales", "Operations", "HR", "Finance", "Customer Support"]
# latent archetype -> (share, dept weights)
ARCH = {
    "star":      (0.18, [.35, .20, .10, .05, .15, .15]),
    "steady":    (0.34, [.20, .15, .25, .10, .15, .15]),
    "burnout":   (0.18, [.25, .25, .10, .03, .07, .30]),
    "disengaged":(0.17, [.10, .20, .25, .08, .12, .25]),
    "newcomer":  (0.13, [.30, .20, .15, .05, .10, .20]),
}
P = {  # mean/sd per archetype
 "age":          dict(star=(36,5), steady=(42,7), burnout=(34,6), disengaged=(40,8), newcomer=(25,3)),
 "tenure":       dict(star=(7,3), steady=(10,4), burnout=(5,2.5), disengaged=(7,4), newcomer=(1,.7)),
 "salary":       dict(star=(95e3,15e3), steady=(70e3,12e3), burnout=(62e3,11e3), disengaged=(50e3,9e3), newcomer=(42e3,6e3)),
 "perf":         dict(star=(4.5,.35), steady=(3.5,.4), burnout=(3.8,.5), disengaged=(2.4,.5), newcomer=(3.1,.5)),
 "engagement":   dict(star=(8.2,.8), steady=(7.2,.9), burnout=(4.5,1.1), disengaged=(3.2,1.0), newcomer=(6.8,1.2)),
 "overtime":     dict(star=(18,6), steady=(8,5), burnout=(42,8), disengaged=(6,4), newcomer=(10,5)),
 "absence":      dict(star=(2,1.5), steady=(4,2), burnout=(10,3), disengaged=(11,3.5), newcomer=(3,2)),
 "training":     dict(star=(48,12), steady=(26,10), burnout=(14,8), disengaged=(8,6), newcomer=(40,10)),
 "since_promo":  dict(star=(1.5,1), steady=(3.5,1.8), burnout=(4,1.5), disengaged=(5,2), newcomer=(0.7,.4)),
 "manager":      dict(star=(4.4,.4), steady=(3.9,.5), burnout=(3.0,.7), disengaged=(2.6,.7), newcomer=(3.8,.6)),
 "projects":     dict(star=(6,1.5), steady=(4,1.3), burnout=(8,1.5), disengaged=(2.5,1.2), newcomer=(2.5,1)),
 "commute":      dict(star=(15,8), steady=(18,10), burnout=(28,12), disengaged=(25,12), newcomer=(14,8)),
}
labels = rng.choice(list(ARCH), size=N, p=[v[0] for v in ARCH.values()])
rows = []
for a in labels:
    r = {k: rng.normal(*v[a]) for k, v in P.items()}
    r["dept"] = rng.choice(DEPTS, p=ARCH[a][1])
    rows.append(r)
df = pd.DataFrame(rows)
df["age"] = df.age.clip(21, 62).round().astype(int)
df["tenure_years"] = df.tenure.clip(0.1, df.age - 20).round(1)
df["monthly_salary"] = (df.salary.clip(25e3, 160e3) / 12).round(-1)
df["performance_rating"] = df.perf.clip(1, 5).round(1)
df["engagement_score"] = df.engagement.clip(1, 10).round(1)
df["overtime_hours_month"] = df.overtime.clip(0, 70).round().astype(int)
df["absenteeism_days"] = df.absence.clip(0, 25).round().astype(int)
df["training_hours"] = df.training.clip(0, 80).round().astype(int)
df["years_since_promotion"] = np.minimum(df.since_promo.clip(0, 12), df.tenure_years).round(1)
df["manager_rating"] = df.manager.clip(1, 5).round(1)
df["projects_handled"] = df.projects.clip(1, 12).round().astype(int)
df["commute_km"] = df.commute.clip(1, 70).round().astype(int)
df.insert(0, "employee_id", [f"E{1000+i}" for i in range(N)])
# historical attrition (validation only; NOT used for clustering)
z = (-1.1 - 0.55*(df.engagement_score-6) + 0.05*(df.overtime_hours_month-15) + 0.12*df.years_since_promotion
     - 0.4*(df.manager_rating-3.5) + 0.03*df.absenteeism_days)
df["left_company"] = (rng.random(N) < 1/(1+np.exp(-z))).astype(int)
cols = ["employee_id","dept","age","tenure_years","monthly_salary","performance_rating","engagement_score",
        "overtime_hours_month","absenteeism_days","training_hours","years_since_promotion",
        "manager_rating","projects_handled","commute_km","left_company"]
df = df[cols].rename(columns={"dept":"department"})
Path(__file__).resolve().parent.parent.joinpath("data").mkdir(exist_ok=True)
df.to_csv(Path(__file__).resolve().parent.parent/"data"/"employees.csv", index=False)
print(df.shape, "attrition rate:", df.left_company.mean().round(3))
