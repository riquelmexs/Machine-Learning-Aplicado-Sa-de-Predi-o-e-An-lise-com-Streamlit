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

st.set_page_config(page_title="Painel Clínico de Machine Learning em Saúde", layout="wide")

st.markdown("""
<style>
body {background: #0f172a;}
.main {background: #0f172a;}
h1,h2,h3,h4,h5,h6 {color: #e2e8f0 !important; font-weight: 500;}
p, label, span {color: #cbd5f5 !important;}
.top-card {background: #111827; border: 1px solid rgba(148,163,184,.12); border-radius: .65rem; padding: 0.85rem 1rem;}
.metric-title {font-size: .72rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .05em;}
.metric-value {font-size: 1.4rem; font-weight: 600; color: #e2e8f0; margin-top: .35rem;}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("Configurações do Estudo")
percentual_teste = st.sidebar.slider("Percentual para teste e validação", 0.1, 0.4, 0.2, 0.05)
semente = st.sidebar.number_input("Semente (reprodutibilidade)", 0, 9999, 42, 1)
kfold = st.sidebar.slider("Validação cruzada (k)", 3, 10, 5, 1)
n_clusters = st.sidebar.slider("Número de grupos (não supervisionado)", 2, 10, 3, 1)

dataset = load_diabetes()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target, name="progresso_da_doenca")

st.title("Painel Clínico — Machine Learning Aplicado à Saúde")
st.write(
    "Este painel demonstra como técnicas de aprendizado de máquina podem auxiliar equipes médicas a **analisar dados clínicos**, "
    "**prever a progressão de uma doença** e **identificar grupos de pacientes com características semelhantes**. "
    "O conjunto de dados utilizado representa informações clínicas de pacientes acompanhados por um ano."
)

c1, c2, c3, c4 = st.columns(4)
c1.markdown(f'<div class="top-card"><div class="metric-title">Pacientes avaliados</div><div class="metric-value">{X.shape[0]}</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="top-card"><div class="metric-title">Variáveis clínicas</div><div class="metric-value">{X.shape[1]}</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="top-card"><div class="metric-title">Desfecho</div><div class="metric-value">progressão da doença</div></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="top-card"><div class="metric-title">Fonte</div><div class="metric-value">Scikit-learn</div></div>', unsafe_allow_html=True)

aba_analise, aba_sup, aba_nao_sup = st.tabs([
    "1️ Análise dos Dados Clínicos",
    "2️ Modelo Supervisionado (Previsão)",
    "3️ Modelo Não Supervisionado (Segmentação)"
])

with aba_analise:
    st.subheader("Análise Exploratória dos Dados")
    st.write(
        "Nesta seção, é possível explorar as variáveis clínicas coletadas. Cada linha representa um paciente, "
        "e cada coluna uma variável clínica ou demográfica. O desfecho é a progressão da doença após um período de acompanhamento."
    )

    st.dataframe(X.head())

    st.subheader("Distribuição das Variáveis Clínicas")
    var_hist = st.selectbox("Selecione uma variável para visualizar sua distribuição", X.columns, index=0)
    fig_hist = px.histogram(X, x=var_hist, nbins=25, template="plotly_dark", title=f"Distribuição de {var_hist}")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Relação Geral com o Desfecho Clínico")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=semente)
    componentes = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(componentes, columns=["Componente 1", "Componente 2"])
    df_pca["progressao"] = y
    fig_pca = px.scatter(
        df_pca,
        x="Componente 1",
        y="Componente 2",
        color="progressao",
        template="plotly_dark",
        title="Projeção dos pacientes (PCA) colorida pelo desfecho"
    )
    st.plotly_chart(fig_pca, use_container_width=True)

    corr = pd.concat([X, y], axis=1).corr(numeric_only=True)
    fig_corr = px.imshow(
        corr,
        template="plotly_dark",
        title="Correlação entre variáveis clínicas e o desfecho"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

with aba_sup:
    st.subheader("Modelo Supervisionado — Previsão da Progressão da Doença")
    st.write(
        "Nesta etapa, o modelo é treinado para **prever a progressão da doença** com base nas variáveis clínicas dos pacientes. "
        "Utiliza-se aprendizado supervisionado, em que o modelo conhece os resultados anteriores (desfechos) e aprende a estimá-los."
    )

    modelo_nome = st.selectbox(
        "Selecione o algoritmo de regressão",
        ["Regressão Linear", "Ridge", "Lasso", "Floresta Aleatória"],
        index=3
    )

    if modelo_nome in ["Ridge", "Lasso"]:
        alpha = st.slider("Intensidade de regularização (alpha)", 0.0001, 10.0, 1.0, 0.1)
    if modelo_nome == "Floresta Aleatória":
        n_estimators = st.slider("Número de árvores", 100, 1000, 400, 50)
        max_depth = st.slider("Profundidade máxima (0 = automática)", 0, 30, 0, 1) or None
        min_samples_split = st.slider("Mínimo para divisão de nó", 2, 10, 2, 1)
        min_samples_leaf = st.slider("Mínimo de amostras por folha", 1, 10, 1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=percentual_teste, random_state=semente
    )

    if modelo_nome in ["Regressão Linear", "Ridge", "Lasso"]:
        scaler = StandardScaler()
        X_train_proc = scaler.fit_transform(X_train)
        X_test_proc = scaler.transform(X_test)
    else:
        X_train_proc, X_test_proc = X_train, X_test

    if modelo_nome == "Regressão Linear":
        modelo = LinearRegression()
    elif modelo_nome == "Ridge":
        modelo = Ridge(alpha=alpha, random_state=semente)
    elif modelo_nome == "Lasso":
        modelo = Lasso(alpha=alpha, random_state=semente, max_iter=10000)
    else:
        modelo = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=semente,
            n_jobs=-1
        )

    modelo.fit(X_train_proc, y_train)
    y_pred = modelo.predict(X_test_proc)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    if modelo_nome in ["Regressão Linear", "Ridge", "Lasso"]:
        cv_scores = cross_val_score(modelo, X_train_proc, y_train, cv=kfold, scoring="r2")
    else:
        cv_scores = cross_val_score(modelo, X_train, y_train, cv=kfold, scoring="r2", n_jobs=-1)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="top-card"><div class="metric-title">MAE (Erro Médio Absoluto)</div><div class="metric-value">{mae:.2f}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="top-card"><div class="metric-title">RMSE (Raiz do Erro Quadrático)</div><div class="metric-value">{rmse:.2f}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="top-card"><div class="metric-title">R² (teste)</div><div class="metric-value">{r2:.3f}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="top-card"><div class="metric-title">R² (média k-fold)</div><div class="metric-value">{cv_scores.mean():.3f}</div></div>', unsafe_allow_html=True)

    st.subheader("Análise de desempenho")
    df_res = pd.DataFrame({"Valor Real": y_test, "Predição": y_pred, "Erro": y_test - y_pred})
    fig_pred = px.scatter(
        df_res,
        x="Valor Real",
        y="Predição",
        template="plotly_dark",
        title="Comparação entre valores reais e previstos"
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    fig_err = px.histogram(
        df_res,
        x="Erro",
        nbins=30,
        template="plotly_dark",
        title="Distribuição dos erros de previsão"
    )
    st.plotly_chart(fig_err, use_container_width=True)

    st.subheader("Variáveis mais relevantes no modelo")
    if hasattr(modelo, "feature_importances_"):
        imp = pd.Series(modelo.feature_importances_, index=X.columns).sort_values(ascending=False)
    elif modelo_nome in ["Regressão Linear", "Ridge", "Lasso"]:
        coef = np.abs(modelo.coef_)
        imp = pd.Series(coef, index=X.columns).sort_values(ascending=False)
    else:
        perm = permutation_importance(
            modelo, X_test_proc, y_test, n_repeats=10, random_state=semente, n_jobs=-1
        )
        imp = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)

    fig_imp = px.bar(
        imp.head(12),
        title="Principais variáveis clínicas utilizadas pelo modelo",
        template="plotly_dark"
    )
    st.plotly_chart(fig_imp, use_container_width=True)

with aba_nao_sup:
    st.subheader("Modelo Não Supervisionado — Agrupamento de Pacientes")
    st.write(
        "O aprendizado não supervisionado busca **identificar padrões e agrupar pacientes com características semelhantes**, "
        "sem utilizar o desfecho clínico. Essa técnica auxilia na criação de perfis clínicos e pode apoiar decisões médicas personalizadas."
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=semente)
    grupos = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, grupos)

    st.markdown(f"**Índice de coerência do agrupamento (Silhouette):** {sil:.3f}")

    pca_seg = PCA(n_components=2, random_state=semente)
    pts = pca_seg.fit_transform(X_scaled)
    df_seg = pd.DataFrame(pts, columns=["Componente 1", "Componente 2"])
    df_seg["Grupo"] = grupos.astype(str)
    fig_seg = px.scatter(
        df_seg,
        x="Componente 1",
        y="Componente 2",
        color="Grupo",
        template="plotly_dark",
        title="Pacientes agrupados por semelhança clínica (PCA 2D)"
    )
    st.plotly_chart(fig_seg, use_container_width=True)

    st.write("**Perfil médio de cada grupo identificado:**")
    df_perf = pd.DataFrame(X, copy=True)
    df_perf["grupo"] = grupos
    medias = df_perf.groupby("grupo").mean(numeric_only=True).round(2)
    st.dataframe(medias)
