import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ---------- Helper functions -----------
def load_data():
    return pd.read_csv('data/leady_veteye_demo.csv')

def train_model(df):
    X = df.drop(columns=['lead_id', 'target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(max_depth=6, n_estimators=200, learning_rate=0.1)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_pred)
    return model, auc, X

# ---------- App layout ----------
def main():
    st.sidebar.image('assets/veteye_logo.png', width=150)
    st.sidebar.title('VetEye AI CRM')
    page = st.sidebar.radio('Nawigacja', ['Dashboard', 'Wszystkie leady', 'Statystyki', 'Ustawienia'])

    if page == 'Dashboard':
        show_dashboard()
    elif page == 'Wszystkie leady':
        show_all_leads()
    elif page == 'Statystyki':
        show_stats()
    elif page == 'Ustawienia':
        show_settings()

# ---------- Pages ----------
def show_dashboard():
    st.title('Top 20 leadów')
    df = load_data()
    model, auc, X = train_model(df)
    df['score'] = model.predict_proba(df.drop(columns=['lead_id','target']))[:,1]
    df['score'] = df['score'].round(2)
    top = df.sort_values('score', ascending=False).head(20)
    st.table(top[['lead_id','score','recommended_action']])

    # Detail view
    lead = st.selectbox('Wybierz leada', top['lead_id'])
    display_lead_detail(df, model)


def show_all_leads():
    st.title('Wszystkie leady')
    df = load_data()
    df['score'] = (xgb.XGBClassifier().fit(df.drop(columns=['lead_id','target']), df['target']).predict_proba(df.drop(columns=['lead_id','target']))[:,1]).round(2)
    sort_feat = st.selectbox('Sortuj po', df.columns.tolist())
    asc = st.checkbox('Rosnąco', value=False)
    st.dataframe(df.sort_values(sort_feat, ascending=asc).head(100))


def show_stats():
    st.title('Statystyki POC')
    df = load_data()
    _, auc, _ = train_model(df)
    st.metric('ROC AUC', f'{auc:.2f}')
    st.bar_chart(df['score'])


def show_settings():
    st.title('Ustawienia')
    st.write('Tu można wgrać dane i ustawić parametry CRM')
    file = st.file_uploader('Wybierz plik CSV z leadami')
    if file:
        df = pd.read_csv(file)
        df.to_csv('data/leady_veteye_demo.csv', index=False)
        st.success('Plik wgrany')

# ---------- Lead detail and explainability ----------
def display_lead_detail(df, model):
    idx = df[df['lead_id']==st.session_state.get('lead_id')].index[0]
    X = df.drop(columns=['lead_id','target'])
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    st.subheader('Feature importance')
    shap.plots.bar(shap_values[idx], show=False)
    st.pyplot(bbox_inches='tight')

    # Rekomendowany skrypt
    action = df.loc[df['lead_id']==st.session_state.get('lead_id'),'recommended_action'].values[0]
    scripts = {
        'call': 'Zadzwoń i zapytaj o... ',
        'email': 'Wyślij mail z propozycją demo na...',
        'offer': 'Przygotuj dedykowaną ofertę dla...'    }
    st.markdown(f"**🗣️ Skrypt rozmowy:** {scripts.get(action,'Skontaktuj się')}" )

if __name__ == '__main__':
    main()
