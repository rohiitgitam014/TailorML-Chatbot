import streamlit as st
import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.metrics import r2_score, classification_report
from fpdf import FPDF
import time

st.set_page_config(page_title="TailorML", page_icon="🧠", layout="centered")

# ======== Styling ========
st.markdown("""
<style>
.chat-input-container label {display: none;}
[data-testid="stSidebar"] {background-color: #F5F7FA;}
[data-testid="stAppViewContainer"] > .main {background-color: #FDFDFD;}
.stButton > button {border-radius: 1rem; padding: 0.5rem 1.2rem; background-color: #4F46E5; color: white;}
</style>
""", unsafe_allow_html=True)

# ======== Missing Value Handling ========
def handle_missing_values(df):
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        missing_ratio = df[col].isnull().mean()
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            imputer = SimpleImputer(strategy='most_frequent')
        else:
            strategy = 'mean' if missing_ratio <= 0.3 else 'median'
            imputer = SimpleImputer(strategy=strategy)
        df[col] = imputer.fit_transform(df[[col]])
    return df

# ======== Model Selector ========
def get_models(task_type):
    if task_type == 'classification':
        return {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "AdaBoost Classifier": AdaBoostClassifier()
        }
    else:
        return {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Gradient Boosting Regressor": GradientBoostingRegressor(),
            "AdaBoost Regressor": AdaBoostRegressor()
        }

# ======== App State ========
st.title("🧠 TailorML: Your Predictive Chat Assistant")

if "stage" not in st.session_state:
    st.session_state.stage = "start"
    st.session_state.df = None

# ======== Stage: Upload ========
if st.session_state.stage == "start":
    st.chat_message("ai").write("👋 Hi, I’m **TailorML**! Let’s predict together. Start by uploading your dataset 📂")
    file = st.file_uploader("Upload your CSV dataset", type=["csv"])
    if file:
        df = pd.read_csv(file)
        df = handle_missing_values(df)
        st.session_state.df = df
        st.session_state.stage = "uploaded"
        st.rerun()

# ======== Stage: Preview ========
elif st.session_state.stage == "uploaded":
    st.chat_message("ai").write("👀 Here's a preview of your dataset:")
    st.dataframe(st.session_state.df.head())
    st.chat_message("ai").write(f"📐 Shape: {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns")

    missing_percent = (st.session_state.df.isnull().sum() / len(st.session_state.df) * 100).round(2)
    if (missing_percent > 0).any():
        st.chat_message("ai").write("🧹 Missing values (%):")
        st.dataframe(missing_percent[missing_percent > 0])

    if st.button("✅ All Good, Next"):
        st.session_state.stage = "choose_target"
        st.rerun()

# ======== Stage: Choose Target ========
elif st.session_state.stage == "choose_target":
    st.chat_message("ai").write("🎯 What would you like to predict? Choose a target column:")
    target = st.selectbox("Select the target column", st.session_state.df.columns)
    if target:
        st.session_state.target = target
        if st.button("🔍 Confirm Target"):
            st.session_state.stage = "predict"
            st.rerun()

# ======== Stage: Predict ========
elif st.session_state.stage == "predict":
    st.chat_message("ai").write("⚙️ Training ML models on your data...")
    time.sleep(1)

    df = st.session_state.df
    target = st.session_state.target
    st.chat_message("ai").write(f"🎯 Target selected: **{target}**")

    X = df.drop(columns=[target])
    y = df[target]

    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col])
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    task_type = "classification" if y.dtype == 'object' or y.nunique() <= 10 else "regression"
    st.chat_message("ai").write(f"📊 Task detected: **{task_type}**")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    models = get_models(task_type)

    best_model = None
    best_score = -np.inf
    best_y_pred = None
    best_model_obj = None
    results = {}

    progress = st.progress(0)
    for i, (name, model) in enumerate(models.items()):
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred) if task_type == "regression" else classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']
            results[name] = round(score, 4)
            if score > best_score:
                best_score = score
                best_model = name
                best_model_obj = model
                best_y_pred = y_pred
        except Exception as e:
            results[name] = f"❌ {str(e)}"
        progress.progress((i + 1) / len(models))

    explanation = {
        "Random Forest": "🌲 Excellent for nonlinear data with lots of features.",
        "Linear Regression": "📈 Great when the relationship between features and target is linear.",
        "KNN": "👥 Works well for small datasets by looking at nearby points.",
        "AdaBoost Classifier": "⚡ Boosts simple learners for better results.",
        "Gradient Boosting Regressor": "🚀 Ideal for structured datasets with patterns."
    }.get(best_model, "✨ A solid pick for your data!")

    st.session_state.update({
        "stage": "results",
        "results": results,
        "best_model": best_model,
        "best_score": best_score,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": best_y_pred,
        "best_model_obj": best_model_obj,
        "explanation": explanation,
        "X_train": X_train,
        "task_type": task_type
    })
    st.rerun()

# ======== Stage: Results ========
elif st.session_state.stage == "results":
    st.chat_message("ai").write(f"🏅 Best model: **{st.session_state.best_model}** | Score: **{st.session_state.best_score:.2f}**")
    st.chat_message("ai").write(f"🤖 Explanation: {st.session_state.explanation}")

    st.chat_message("ai").write("📈 All model performances:")
    st.dataframe(pd.DataFrame.from_dict(st.session_state.results, orient='index', columns=['Score']))

    df_preds = st.session_state.X_test.copy()
    df_preds['Actual'] = st.session_state.y_test
    df_preds['Predicted'] = st.session_state.y_pred

    csv = df_preds.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")

    def generate_pdf_report(df, model_name, score, shap_fig=None, lime_fig=None, user_notes=None, lime_explanations=[]):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt="TailorML Prediction Report", ln=True, align="C")
        pdf.ln(10)

        pdf.cell(200, 10, txt=f"Best Model: {model_name}", ln=True)
        pdf.cell(200, 10, txt=f"Model Score: {score:.4f}", ln=True)
        pdf.ln(10)

        if user_notes:
            pdf.set_font("Arial", style="B", size=12)
            pdf.cell(200, 10, txt="User Comments / Recommendations:", ln=True)
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, txt=user_notes)
            pdf.ln(10)

        pdf.cell(200, 10, txt="Sample Predictions:", ln=True)
        for i in range(min(10, len(df))):
            row = df.iloc[i]
            row_str = ", ".join([f"{col}: {val}" for col, val in row.items()])
            pdf.multi_cell(0, 10, txt=row_str)

        if shap_fig:
            shap_fig.savefig("shap_plot.png")
            pdf.image("shap_plot.png", x=10, y=pdf.get_y(), w=180)
            pdf.ln(85)

        if lime_fig:
            lime_fig.savefig("lime_plot.png")
            pdf.image("lime_plot.png", x=10, y=pdf.get_y(), w=180)
            pdf.ln(85)

        for i, lime_fig_i in enumerate(lime_explanations):
            lime_fig_i.savefig(f"lime_plot_{i}.png")
            pdf.image(f"lime_plot_{i}.png", x=10, y=pdf.get_y(), w=180)
            pdf.ln(85)

        pdf.output("TailorML_Report.pdf")

    if st.button("📄 Export PDF Report"):
        generate_pdf_report(df_preds, st.session_state.best_model, st.session_state.best_score)
        with open("TailorML_Report.pdf", "rb") as f:
            st.download_button("⬇️ Download PDF Report", f, file_name="TailorML_Report.pdf")

    if hasattr(st.session_state.best_model_obj, 'feature_importances_'):
        st.subheader("🔍 Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': st.session_state.X_train.columns,
            'Importance': st.session_state.best_model_obj.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        st.bar_chart(importance_df.set_index('Feature'))

    try:
        explainer = shap.Explainer(st.session_state.best_model_obj, st.session_state.X_train)
        shap_values = explainer(st.session_state.X_test[:100])
        st.subheader("🔬 SHAP Explanation Summary")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, st.session_state.X_test[:100], show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"SHAP explanation skipped: {e}")

    try:
        st.subheader("💡 LIME Explanation (1st test instance)")
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(st.session_state.X_train),
            feature_names=st.session_state.X_train.columns.tolist(),
            class_names=["output"],
            mode=st.session_state.task_type
        )
        exp = lime_explainer.explain_instance(
            st.session_state.X_test.iloc[0].values,
            st.session_state.best_model_obj.predict,
            num_features=10
        )
        fig = exp.as_pyplot_figure()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"LIME explanation skipped: {e}")

    if st.button("🔁 Try New Dataset"):
        st.session_state.stage = "start"
        st.rerun()

# ======== Chat Prompt (Optional) ========
user_input = st.chat_input("💬 Ask TailorML about your results or next steps...")
if user_input:
    st.chat_message("user").write(user_input)
    st.chat_message("ai").write("I'm still learning to chat back! But I'm happy to guide you through another ML task 🚀")
