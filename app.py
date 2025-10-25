import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
import plotly.express as px

st.set_page_config(page_title="ML em Saúde — Diabetes (sklearn)", layout="wide")

st.sidebar.title("⚙️ Configurações")
dataset_choice = st.sidebar.selectbox("Selecione o conjunto de dados", ["Diabetes (sklearn)", "Enviar CSV próprio"])
test_size = st.sidebar.slider("Proporção de dados de teste", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Semente aleatória", min_value=0, value=42, step=1)
cv_folds = st.sidebar.slider("Folds de validação cruzada", 3, 10, 5, 1)
st.sidebar.markdown("---")
st.sidebar.caption("Aplicação para apresentação acadêmica de ML aplicado à saúde.")

def carregar_diabetes():
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="progressao_doenca")
    desc = "Conjunto de dados 'Diabetes' do scikit-learn com 10 preditores normalizados e alvo contínuo representando progressão da doença após 1 ano."
    return X, y, desc

def carregar_csv():
    file = st.sidebar.file_uploader("Envie um arquivo CSV", type=["csv"])
    if file is None:
        return None, None, None
    df = pd.read_csv(file)
    st.session_state["_uploaded_df"] = df
    return df, None, "CSV enviado pelo usuário."

if dataset_choice == "Diabetes (sklearn)":
    X, y, ds_desc = carregar_diabetes()
    target_name = y.name
else:
    df, _, ds_desc = carregar_csv()
    if df is not None:
        cols = list(df.columns)
        target_name = st.sidebar.selectbox("Selecione a coluna alvo (regressão)", cols, index=len(cols)-1 if len(cols)>0 else 0)
        y = df[target_name] if target_name in df else None
        X = df.drop(columns=[target_name]) if target_name in df else None
    else:
        X, y, target_name = None, None, None

st.title("🏥 Machine Learning em Saúde — Demonstração Interativa")
st.write("Aplicação interativa para **aprendizado supervisionado (regressão)** e **não supervisionado (agrupamento)** em dados de saúde.")

if X is None or y is None:
    st.info("Envie um CSV no menu lateral ou selecione o conjunto Diabetes para começar.")
    st.stop()

with st.expander("ℹ️ Sobre o conjunto de dados", expanded=True):
    st.write(ds_desc)
    st.write(f"**Amostras:** {X.shape[0]} — **Atributos:** {X.shape[1]} — **Alvo:** `{target_name}`")
    st.dataframe(X.head())

tab_eda, tab_sup, tab_unsup = st.tabs(["🔎 EDA (Exploração)", "🎯 Supervisionado — Regressão", "🧩 Não Supervisionado — Clusters"])

with tab_eda:
    st.subheader("Distribuições e Correlações")
    col1, col2 = st.columns(2)
    with col1:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=2, random_state=random_state)
        comp = pca.fit_transform(X_scaled)
        df_pca = pd.DataFrame(comp, columns=["PC1", "PC2"])
        df_pca[target_name] = y.values
        fig = px.scatter(df_pca, x="PC1", y="PC2", color=target_name, title="Projeção PCA (colorido pelo alvo)")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        corr = pd.concat([X, y], axis=1).corr(numeric_only=True)
        fig_corr = px.imshow(corr, title="Matriz de Correlação", aspect="auto")
        st.plotly_chart(fig_corr, use_container_width=True)
    st.caption("Dica: avance para as abas de Modelos para comparar algoritmos e métricas.")

with tab_sup:
    st.subheader("Treinamento e Avaliação de Modelos")
    model_name = st.selectbox("Modelo de Regressão", ["LinearRegression", "Ridge", "Lasso", "RandomForestRegressor"])
    if model_name in ["Ridge", "Lasso"]:
        alpha = st.slider("Parâmetro de regularização (alpha)", 0.0001, 10.0, 1.0, 0.1)
    if model_name == "RandomForestRegressor":
        n_estimators = st.slider("Número de árvores (n_estimators)", 100, 1000, 400, 50)
        max_depth = st.slider("Profundidade máxima (0 = None)", 0, 30, 0, 1) or None
        min_samples_split = st.slider("Mín. amostras para split", 2, 10, 2, 1)
        min_samples_leaf = st.slider("Mín. amostras por folha", 1, 10, 1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    if model_name in ["LinearRegression", "Ridge", "Lasso"]:
        scaler = StandardScaler()
        X_train_proc = scaler.fit_transform(X_train)
        X_test_proc = scaler.transform(X_test)
    else:
        X_train_proc, X_test_proc = X_train, X_test
    if model_name == "LinearRegression":
        model = LinearRegression()
    elif model_name == "Ridge":
        model = Ridge(alpha=alpha, random_state=random_state)
    elif model_name == "Lasso":
        model = Lasso(alpha=alpha, random_state=random_state, max_iter=10000)
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=random_state, n_jobs=-1)
    model.fit(X_train_proc, y_train)
    preds = model.predict(X_test_proc)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    if model_name in ["LinearRegression", "Ridge", "Lasso"]:
        cv_scores = cross_val_score(model, X_train_proc, y_train, cv=cv_folds, scoring="r2")
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="r2", n_jobs=-1)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE (Erro Absoluto Médio)", f"{mae:.2f}")
    c2.metric("RMSE (Raiz do Erro Quadrático)", f"{rmse:.2f}")
    c3.metric("R² (teste)", f"{r2:.3f}")
    c4.metric("R² (CV média)", f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    col_a, col_b = st.columns(2)
    with col_a:
        df_res = pd.DataFrame({"Valor Real": y_test, "Predição": preds, "Erro": y_test - preds})
        fig_pred = px.scatter(df_res, x="Valor Real", y="Predição", trendline="ols", title="Verdadeiro vs. Predito")
        st.plotly_chart(fig_pred, use_container_width=True)
    with col_b:
        fig_res = px.histogram(df_res, x="Erro", nbins=30, title="Distribuição dos Erros")
        st.plotly_chart(fig_res, use_container_width=True)
    st.markdown("##### Importância dos Atributos / Coeficientes")
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    elif model_name in ["LinearRegression", "Ridge", "Lasso"]:
        coef = np.abs(model.coef_)
        imp = pd.Series(coef, index=X.columns).sort_values(ascending=False)
    else:
        perm = permutation_importance(model, X_test_proc, y_test, n_repeats=10, random_state=random_state, n_jobs=-1)
        imp = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
    fig_imp = px.bar(imp.head(15), title="Top Atributos")
    st.plotly_chart(fig_imp, use_container_width=True)

with tab_unsup:
    st.subheader("Agrupamento (K-Means) e Projeção PCA")
    n_clusters = st.slider("Número de grupos (k)", 2, 10, 3, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    st.metric("Silhouette Score", f"{sil:.3f}")
    pca = PCA(n_components=2, random_state=random_state)
    points = pca.fit_transform(X_scaled)
    df_k = pd.DataFrame(points, columns=["PC1", "PC2"])
    df_k["cluster"] = labels.astype(str)
    fig_k = px.scatter(df_k, x="PC1", y="PC2", color="cluster", title="Clusters (PCA 2D)")
    st.plotly_chart(fig_k, use_container_width=True)
    prof = pd.DataFrame(X, copy=True)
    prof["cluster"] = labels
    prof_mean = prof.groupby("cluster").mean(numeric_only=True)
    st.dataframe(prof_mean)