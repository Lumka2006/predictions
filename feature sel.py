import os, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, joblib, warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, multilabel_confusion_matrix
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from docx import Document
warnings.filterwarnings("ignore")

# Load Data
data_path, target_col = "data.xlsx", "Target"
if not os.path.exists(data_path): raise FileNotFoundError(f"{data_path} not found!")
df = pd.read_excel(data_path)
if target_col not in df: raise ValueError(f"Missing target '{target_col}'!")
print(f"✅ Data loaded: {df.shape}")

# Encode Categoricals
encoders = {}
for col in df.select_dtypes('object'):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    if col == target_col: print("Target encoding:", dict(zip(le.classes_, le.transform(le.classes_))))

X, y = df.drop(target_col, axis=1), df[target_col]

# Feature Importance
anova = SelectKBest(f_classif, k='all').fit(X, y)
rf = RandomForestClassifier(random_state=42).fit(X, y)
fi = pd.DataFrame({"ANOVA": anova.scores_, "RF": rf.feature_importances_}, index=X.columns)
fi = fi.div(fi.max()).assign(Overall=lambda d: d.mean(1)).sort_values("Overall", ascending=False)

with pd.ExcelWriter("FeatureImportance.xlsx") as w:
    pd.Series(anova.scores_, index=X.columns).sort_values(ascending=False).to_excel(w, "ANOVA")
    pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).to_excel(w, "RF")
    fi.to_excel(w, "Combined")

top_features = fi.head(15).index.tolist()
df[top_features + [target_col]].to_excel("SelectedFeatures.xlsx", index=False)

fi["Overall"].plot.barh(color="skyblue"); plt.gca().invert_yaxis(); plt.show()

# Clean Data
df = pd.read_excel("SelectedFeatures.xlsx")
df.columns = df.columns.str.strip().str.replace(r"[() ]", "_", regex=True)
num_cols, cat_cols = df.select_dtypes(np.number).columns, df.select_dtypes(exclude=np.number).columns
df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
if len(cat_cols): df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])
print("✅ Data cleaned")

# Outliers
outliers = df[(np.abs(stats.zscore(df[num_cols])) > 3).any(1)]
outliers.to_excel("outliers.xlsx", index=False)
print(f"📊 {len(outliers)} outliers saved")

# Plots
sns.set_style("whitegrid")
for c in num_cols:
    plt.hist(df[c], bins=15, color='aqua', edgecolor='black', alpha=0.7)
    mean, med, mode = df[c].mean(), df[c].median(), df[c].mode()[0]
    plt.gcf().text(0.75,0.7,f'Mean:{mean:.2f}\nMedian:{med:.2f}\nMode:{mode:.2f}', fontsize=9, bbox=dict(facecolor='white', alpha=0.6))
    plt.title(f'{c} Distribution'); plt.tight_layout(); plt.show()
for c in cat_cols:
    sns.countplot(x=c, data=df, palette='Set3', order=df[c].value_counts().index)
    plt.title(c); plt.xticks(rotation=45); plt.show()

# ANOVA
with pd.ExcelWriter("anova_results.xlsx") as w:
    for c in num_cols[:15]:
        sm.stats.anova_lm(ols(f"Q(\"{c}\") ~ C({target_col})", data=df).fit(), typ=2).to_excel(w, sheet_name=c[:31])
print("✅ ANOVA results saved")

# Train/Test Split
X, y = df.drop(target_col, axis=1), df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train.assign(Target=y_train).to_excel("train_data.xlsx", index=False)
X_test.assign(Target=y_test).to_excel("test_data.xlsx", index=False)

# Train RF + Predict
rf_final = RandomForestClassifier(n_estimators=300, random_state=42).fit(X_train, y_train)
y_pred = rf_final.predict(X_test)
pd.concat([X_test, pd.Series(y_pred, name=target_col)], axis=1).to_excel("predicted_results.xlsx", index=False)

# Evaluation Report
labels = rf_final.classes_
mcm = multilabel_confusion_matrix(y_test, y_pred, labels=labels)
cm = confusion_matrix(y_test, y_pred, labels=labels)

doc = Document()
doc.add_heading("🎓 Model Evaluation Report", 1)
doc.add_heading("Overall Performance", 2)
rep = classification_report(y_test, y_pred, output_dict=True)
acc = accuracy_score(y_test, y_pred)

t = doc.add_table(rows=3, cols=5); t.style="Light Grid Accent 1"
for i,h in enumerate(["Metric","Accuracy","Precision","Recall","F1"]): t.cell(0,i).text = h
t.cell(1,0).text, t.cell(1,1).text = "Weighted Avg", f"{acc:.4f}"
t.cell(1,2).text, t.cell(1,3).text, t.cell(1,4).text = f"{rep['weighted avg']['precision']:.4f}", f"{rep['weighted avg']['recall']:.4f}", f"{rep['weighted avg']['f1-score']:.4f}"
t.cell(2,0).text, t.cell(2,1).text = "Macro Avg", "-"
t.cell(2,2).text, t.cell(2,3).text, t.cell(2,4).text = f"{rep['macro avg']['precision']:.4f}", f"{rep['macro avg']['recall']:.4f}", f"{rep['macro avg']['f1-score']:.4f}"

doc.add_heading("Overall Confusion Matrix", 2)
cm_table = doc.add_table(rows=cm.shape[0]+1, cols=cm.shape[1]+1); cm_table.style="Light Grid Accent 1"
cm_table.cell(0,0).text = "Actual/Predicted"
for j,l in enumerate(labels): cm_table.cell(0,j+1).text = str(l)
for i,l in enumerate(labels):
    cm_table.cell(i+1,0).text = str(l)
    for j in range(cm.shape[1]): cm_table.cell(i+1,j+1).text = str(cm[i,j])

# Per-Class Details
doc.add_heading("Per-Class Details", 2)

for class_index, class_label in enumerate(labels):
    # Confusion matrix values
    tn, fp, fn, tp = mcm[class_index].ravel()
    total_class_samples = tp + tn + fp + fn
    
    # Metrics
    class_accuracy = (tp + tn) / total_class_samples if total_class_samples else 0
    class_error = 1 - class_accuracy
    class_precision = tp / (tp + fp) if (tp + fp) else 0
    class_recall = tp / (tp + fn) if (tp + fn) else 0
    class_specificity = tn / (tn + fp) if (tn + fp) else 0
    class_f1 = (2 * class_precision * class_recall) / (class_precision + class_recall) if (class_precision + class_recall) else 0

    # Heading for this class
    doc.add_heading(f"🔹 Class: {class_label}", 3)

    # Metric formulas
    for metric_description in [
        "Accuracy=(TP+TN)/(TP+TN+FP+FN)",
        "ErrorRate=1-Accuracy",
        "Precision=TP/(TP+FP)",
        "Recall=TP/(TP+FN)",
        "Specificity=TN/(TN+FP)",
        "F1=2*(Prec*Rec)/(Prec+Rec)"
    ]:
        doc.add_paragraph(f"• {metric_description}", style="List Bullet")

    # Metric values
    doc.add_paragraph(
        f"Acc=({tp}+{tn})/({tp}+{tn}+{fp}+{fn})={class_accuracy:.4f}\n"
        f"Err=1-{class_accuracy:.4f}={class_error:.4f}\n"
        f"Prec={tp}/({tp}+{fp})={class_precision:.4f}\n"
        f"Rec={tp}/({tp}+{fn})={class_recall:.4f}\n"
        f"Spec={tn}/({tn}+{fp})={class_specificity:.4f}\n"
        f"F1=2*({class_precision:.4f}×{class_recall:.4f})/({class_precision:.4f}+{class_recall:.4f})={class_f1:.4f}"
    )
    # Confusion table
    class_table = doc.add_table(rows=3, cols=3)
    class_table.style = "Light Grid Accent 1"
    class_table.cell(0,1).text, class_table.cell(0,2).text = "Pred:Position", "Pred:Negative"
    class_table.cell(1,0).text, class_table.cell(2,0).text = "Actual:Position", "Actual:Neg"
    class_table.cell(1,1).text, class_table.cell(1,2).text = str(tp), str(fn)
    class_table.cell(2,1).text, class_table.cell(2,2).text = str(fp), str(tn)

    doc.add_paragraph(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")

doc.save("model_evaluation.docx")
joblib.dump(rf_final, "optimized_rf_model.pkl")
print("✅ Report & Model saved")
