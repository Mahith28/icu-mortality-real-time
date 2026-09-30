import os
import json
import numpy as np
import pandas as pd
BASE_DIR="/home/mahith/BDA_PROJECT/Model/ensemble"
RESULTS_DIR=os.path.join(BASE_DIR,"results")
INPUT_FILE=os.path.join(RESULTS_DIR,"ensemble_threshold_search.csv")
LOCKED_FILE=os.path.join(RESULTS_DIR,"locked_ensemble_threshold.json")
OUTPUT_TABLE=os.path.join(RESULTS_DIR,"operating_point_table.csv")
OUTPUT_SELECTED=os.path.join(RESULTS_DIR,"operating_point_selected_rows.csv")
OUTPUT_REPORT=os.path.join(RESULTS_DIR,"operating_point_analysis.txt")
LOCKED_TOLERANCE=1e-9
required={"threshold","sensitivity","specificity","precision","npv","f1","f2","accuracy","alert_rate","tp","fp","tn","fn"}
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Missing threshold search file: {INPUT_FILE}")
if not os.path.exists(LOCKED_FILE):
    raise FileNotFoundError(f"Missing locked threshold file: {LOCKED_FILE}")
df=pd.read_csv(INPUT_FILE)
missing=required-set(df.columns)
if missing:
    raise RuntimeError(f"Missing required columns: {sorted(missing)}")
if df.empty:
    raise RuntimeError("Threshold search file is empty")
numeric_columns=list(required)
for column in numeric_columns:
    df[column]=pd.to_numeric(df[column],errors="raise")
if not np.isfinite(df[numeric_columns].to_numpy(dtype=np.float64)).all():
    raise RuntimeError("Non-finite values found in threshold search")
if not ((df["threshold"]>=0)&(df["threshold"]<=1)).all():
    raise RuntimeError("Threshold values outside [0,1]")
for column in ["sensitivity","specificity","precision","npv","f1","f2","accuracy","alert_rate"]:
    if not ((df[column]>=0)&(df[column]<=1)).all():
        raise RuntimeError(f"{column} contains values outside [0,1]")
if (df["tp"]<0).any() or (df["fp"]<0).any() or (df["tn"]<0).any() or (df["fn"]<0).any():
    raise RuntimeError("Negative confusion-matrix counts found")
if df["threshold"].duplicated().any():
    raise RuntimeError("Duplicate thresholds found")
df=df.sort_values("threshold").reset_index(drop=True)
with open(LOCKED_FILE,"r") as f:
    locked=json.load(f)
locked_threshold=float(locked["threshold"])
locked_sensitivity=float(locked["sensitivity"])
locked_specificity=float(locked["specificity"])
locked_alert_rate=float(locked["alert_rate"])
threshold_match=np.isclose(
    df["threshold"].to_numpy(),
    locked_threshold,
    atol=LOCKED_TOLERANCE,
    rtol=0
)
if not threshold_match.any():
    raise RuntimeError(f"Locked threshold {locked_threshold:.6f} not found in threshold search")
locked_rows=df.loc[threshold_match].copy()
if len(locked_rows)!=1:
    raise RuntimeError("Locked threshold does not map to exactly one threshold row")
locked_row=locked_rows.iloc[0]
if not np.isclose(locked_row["sensitivity"],locked_sensitivity,atol=1e-6):
    raise RuntimeError("Locked sensitivity does not match threshold search")
if not np.isclose(locked_row["specificity"],locked_specificity,atol=1e-6):
    raise RuntimeError("Locked specificity does not match threshold search")
if not np.isclose(locked_row["alert_rate"],locked_alert_rate,atol=1e-6):
    raise RuntimeError("Locked alert rate does not match threshold search")
df["locked_operating_point"]=np.isclose(
    df["threshold"],
    locked_threshold,
    atol=LOCKED_TOLERANCE,
    rtol=0
)
selected_thresholds=[0.05,0.075,0.10,0.108,0.125,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80,0.90]
threshold_values=df["threshold"].to_numpy()
selected_indices=[]
for threshold in selected_thresholds:
    selected_indices.append(np.abs(threshold_values-threshold).argmin())
selected=df.iloc[selected_indices].copy()
selected=selected.drop_duplicates(subset=["threshold"]).sort_values("threshold").reset_index(drop=True)
if not np.isclose(
    selected["threshold"],
    locked_threshold,
    atol=LOCKED_TOLERANCE,
    rtol=0
).any():
    selected=pd.concat(
        [selected,locked_row.to_frame().T],
        ignore_index=True
    )
selected=selected.drop_duplicates(subset=["threshold"]).sort_values("threshold").reset_index(drop=True)
selected["locked_operating_point"]=np.isclose(
    selected["threshold"],
    locked_threshold,
    atol=LOCKED_TOLERANCE,
    rtol=0
)
output_columns=[
    "threshold",
    "sensitivity",
    "specificity",
    "precision",
    "npv",
    "f1",
    "f2",
    "accuracy",
    "alert_rate",
    "tp",
    "fp",
    "tn",
    "fn",
    "locked_operating_point"
]
df[output_columns].to_csv(OUTPUT_TABLE,index=False)
selected[output_columns].to_csv(OUTPUT_SELECTED,index=False)
with open(OUTPUT_REPORT,"w") as f:
    f.write("Operating-point trade-off analysis\n")
    f.write(f"Threshold search rows: {len(df)}\n")
    f.write(f"Locked threshold: {locked_threshold:.6f}\n")
    f.write(f"Locked sensitivity: {locked_sensitivity:.6f}\n")
    f.write(f"Locked specificity: {locked_specificity:.6f}\n")
    f.write(f"Locked alert rate: {locked_alert_rate:.6f}\n")
    f.write("\n")
    f.write("The locked threshold was not changed by this analysis.\n")
    f.write("This phase is descriptive and is not used for threshold selection.\n")
    f.write("No test data were used.\n")
    f.write("Selected operating points\n")
    for _,row in selected.iterrows():
        marker=" [LOCKED]" if row["locked_operating_point"] else ""
        f.write(
            f"Threshold={row['threshold']:.3f}{marker}, "
            f"Sensitivity={row['sensitivity']:.6f}, "
            f"Specificity={row['specificity']:.6f}, "
            f"PPV={row['precision']:.6f}, "
            f"NPV={row['npv']:.6f}, "
            f"F1={row['f1']:.6f}, "
            f"F2={row['f2']:.6f}, "
            f"Accuracy={row['accuracy']:.6f}, "
            f"Alert rate={row['alert_rate']:.6f}\n"
        )
print(f"Loaded threshold search rows: {len(df)}")
print(f"Locked threshold: {locked_threshold:.3f}")
print(f"Locked sensitivity: {locked_sensitivity:.6f}")
print(f"Locked specificity: {locked_specificity:.6f}")
print(f"Locked alert rate: {locked_alert_rate:.6f}")
print(f"Selected operating points: {len(selected)}")
print(f"Operating-point table saved: {OUTPUT_TABLE}")
print(f"Selected operating points saved: {OUTPUT_SELECTED}")
print(f"Operating-point report saved: {OUTPUT_REPORT}")
print("Operating-point analysis completed successfully")