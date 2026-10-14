import pandas as pd, plotly.graph_objects as go, plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
R = Path(__file__).resolve().parent.parent
d = pd.read_csv(R/"data/employees_segmented.csv")
order = ["High Performers","Steady Core","New Joiners","Burnout Risk","Disengaged Low Performers"]
col = dict(zip(order, ["#59A14F","#4C78A8","#76B7B2","#F28E2B","#E15759"]))
cfg = dict(displaylogo=False, responsive=True)
def fig_html(f, first=False): return f.to_html(full_html=False, include_plotlyjs=True if first else False, config=cfg)
lay = dict(template="plotly_white", margin=dict(l=40,r=20,t=50,b=40), height=380, font=dict(family="Inter, Arial"))

seg = d.groupby("segment").agg(n=("employee_id","count"), attr=("left_company","mean"), perf=("performance_rating","mean"),
        eng=("engagement_score","mean"), ot=("overtime_hours_month","mean"), hr=("high_risk_flag","sum")).loc[order].reset_index()
f1 = go.Figure(go.Bar(x=seg.segment, y=seg.n, marker_color=[col[s] for s in seg.segment], text=seg.n, textposition="outside"))
f1.update_layout(title="Headcount by segment", **lay)
f2 = px.scatter(d, x="PC1", y="PC2", color="segment", color_discrete_map=col, category_orders={"segment":order}, hover_data=["employee_id","department","performance_rating","engagement_score"], opacity=.7)
f2.update_layout(title="Workforce map (PCA)", **lay)
f3 = go.Figure(go.Bar(y=seg.segment[::-1], x=(seg.attr*100)[::-1], orientation="h", marker_color=[col[s] for s in seg.segment[::-1]], text=(seg.attr*100).round(0).astype(int).astype(str)[::-1]+"%", textposition="outside"))
f3.add_vline(x=d.left_company.mean()*100, line_dash="dash"); f3.update_layout(title="Retention risk: attrition % by segment", xaxis_title="%", xaxis_range=[0,100], **lay)
ct = (pd.crosstab(d.department, d.segment, normalize="index")*100).loc[:, order]
f4 = go.Figure([go.Bar(name=s, x=ct.index, y=ct[s], marker_color=col[s]) for s in order]); f4.update_layout(barmode="stack", title="Segment mix by department (%)", **lay)
f5 = go.Figure(go.Scatter(x=seg.ot, y=seg.eng, mode="markers+text", text=seg.segment.str.replace(" ","<br>",1), textposition="top center", marker=dict(size=seg.n/4, color=[col[s] for s in seg.segment], opacity=.8)))
f5.update_layout(title="Risk matrix: workload vs engagement", xaxis_title="Avg overtime hrs/month", yaxis_title="Avg engagement", xaxis_range=[0,52], yaxis_range=[1,10], **lay)
pg = pd.crosstab(d.segment, d.performance_group, normalize="index").loc[order, ["Low (<3)","Solid (3–3.9)","Top (≥4)"]]*100
f6 = go.Figure([go.Bar(name=c, x=pg.index, y=pg[c], marker_color=m) for c, m in zip(pg.columns, ["#E15759","#BAB0AC","#59A14F"])]); f6.update_layout(barmode="stack", title="Performance groups by segment (%)", **lay)
dep = d.groupby("department").agg(hc=("employee_id","count"), risk=("high_risk_flag","mean"), attr=("left_company","mean")).sort_values("risk", ascending=False)*[1,100,100]
f7 = go.Figure([go.Bar(name="High-risk share %", x=dep.index, y=dep.risk, marker_color="#F28E2B"), go.Bar(name="Attrition %", x=dep.index, y=dep.attr, marker_color="#E15759")]); f7.update_layout(barmode="group", title="Department hot-spots", **lay)

hr = d[d.high_risk_flag==1].sort_values("risk_score", ascending=False).head(15)[["employee_id","department","segment","overtime_hours_month","absenteeism_days","engagement_score","manager_rating","risk_score"]].round(2)
table = hr.to_html(index=False, classes="t", border=0)
kpi = [("Employees", f"{len(d):,}"), ("Segments", "5"), ("Attrition", f"{d.left_company.mean():.0%}"),
       ("High-risk employees", f"{int(d.high_risk_flag.sum())}"), ("Burnout Risk attrition", f"{seg.set_index('segment').loc['Burnout Risk','attr']:.0%}")]
kp = "".join(f'<div class="k"><span>{a}</span><b>{b}</b></div>' for a,b in kpi)
figs = [f1,f2,f3,f4,f5,f6,f7]
body = "".join(f'<div class="c">{fig_html(f, i==0)}</div>' for i,f in enumerate(figs))
html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Workforce Analytics Dashboard</title><style>
body{{font-family:Inter,Arial,sans-serif;background:#f5f6f8;margin:0;color:#1f2430}}header{{padding:22px 28px;background:#1f2430;color:#fff}}
header h1{{margin:0;font-size:22px}}header p{{margin:4px 0 0;opacity:.7;font-size:13px}}main{{padding:20px 28px;max-width:1300px;margin:auto}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:16px}}
.k{{background:#fff;border-radius:10px;padding:14px 16px;box-shadow:0 1px 3px #0001}}.k span{{display:block;font-size:12px;color:#667}}.k b{{font-size:26px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:14px}}.c{{min-width:0;background:#fff;border-radius:10px;padding:6px;box-shadow:0 1px 3px #0001;overflow:hidden}}
h2{{font-size:16px;margin:22px 0 8px}}.t{{border-collapse:collapse;width:100%;background:#fff;font-size:13px}}.t th,.t td{{padding:7px 10px;border-bottom:1px solid #eee;text-align:left}}.t th{{background:#fafafa}}
@media(max-width:560px){{.grid{{grid-template-columns:1fr}}main{{padding:14px}}}}</style></head><body>
<header><h1>Workforce Analytics Dashboard</h1><p>Employee segmentation via PCA + K-Means · 1,200 synthetic employees</p></header>
<main><div class="kpis">{kp}</div><div class="grid">{body}</div><h2>Top 15 highest-risk employees</h2><div style="overflow-x:auto">{table}</div></main><script>window.addEventListener('load',()=>{{setTimeout(()=>window.dispatchEvent(new Event('resize')),300)}});</script></body></html>"""
(R/"dashboard"/"workforce_dashboard.html").write_text(html)
print("ok", len(html)//1024, "KB")
