import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc, mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="ML Saúde - Streamlit", page_icon="", layout="wide")

st.title(" Machine Learning na Área da Saúde")
st.write("""
Aplicação interativa em **Streamlit** demonstrando **Aprendizagem Supervisionada e Não Supervisionada** 
em datasets da área da saúde. Você pode usar um dataset embutido (ex.: Câncer de Mama, Diabetes) 
ou carregar seu próprio CSV.
""")


st.sidebar.header(" Configurações")
data_source = st.sidebar.radio("Fonte de dados", ["Dataset embutido (sklearn)", "Carregar CSV"])

def load_sklearn_dataset(name: str):
    if name == "Câncer de Mama (classificação)":
        data = datasets.load_breast_cancer(as_frame=True)
        df = data.frame.copy()
        target_col = data.target.name if hasattr(data.target, "name") else "target"
        df.rename(columns={target_col: "target"}, inplace=True)
        return df, "target", "classification", data
    elif name == "Diabetes (regressão)":
        data = datasets.load_diabetes(as_frame=True)
        df = data.frame.copy()
        target_col = data.target.name if hasattr(data.target, "name") else "target"
        df.rename(columns={target_col: "target"}, inplace=True)
        return df, "target", "regression", data
    else:
        st.stop()

if data_source == "Dataset embutido (sklearn)":
    dataset_name = st.sidebar.selectbox("Escolha o dataset", ["Câncer de Mama (classificação)", "Diabetes (regressão)"])
    df, target_col, task_type, raw = load_sklearn_dataset(dataset_name)
    st.success(f"Dataset selecionado: **{dataset_name}** — amostras: {df.shape[0]}, features: {df.shape[1]-1}")
else:
    uploaded = st.sidebar.file_uploader("Envie um CSV (inclua a coluna alvo, se quiser supervisionado)", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        
        candidate_targets = [c for c in df.columns if c.lower() in ["target","label","y","classe","class","diagnosis"]]
        target_col = st.sidebar.selectbox("Coluna alvo (opcional)", ["<nenhuma>"] + list(df.columns), index=(0 if not candidate_targets else (["<nenhuma>"]+list(df.columns)).index(candidate_targets[0])) )
        task_type = "auto"
        if target_col == "<nenhuma>":
            target_col = None
            st.info("Nenhuma coluna alvo selecionada. Você pode usar **Não Supervisionado** (K-Means/PCA) ou escolher manualmente uma coluna alvo acima.")
        else:
            
            if pd.api.types.is_numeric_dtype(df[target_col]) and df[target_col].nunique() > 10:
                task_type = "regression"
            else:
                task_type = "classification"
        st.success(f"CSV carregado: {df.shape[0]} linhas × {df.shape[1]} colunas. Tarefa: {task_type if target_col else 'sem alvo (usar não supervisionado)'}")
    else:
        st.warning("Envie um CSV ou selecione um dataset embutido para começar.")
        st.stop()


non_numeric_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c != (target_col or "")]
if non_numeric_cols:
    st.sidebar.write(" Colunas não numéricas detectadas:", non_numeric_cols)
    handle_cat = st.sidebar.selectbox("Tratamento de não numéricas", ["Ignorar (remover)", "One-hot encoding (experimental)"], index=0)
    if handle_cat == "Ignorar (remover)":
        df = df.drop(columns=non_numeric_cols)
    else:
        df = pd.get_dummies(df, columns=non_numeric_cols, drop_first=True)


constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
if constant_cols:
    df = df.drop(columns=constant_cols)

st.markdown("###  Visão Geral do Dataset")
colA, colB, colC = st.columns(3)
with colA:
    st.metric("Amostras", df.shape[0])
with colB:
    st.metric("Features (X)", df.shape[1] - (1 if target_col else 0))
with colC:
    st.metric("Tem coluna alvo?", "Sim" if target_col else "Não")

with st.expander("👀 Prévia dos dados"):
    st.dataframe(df.head(20), use_container_width=True)


tab1, tab2, tab3, tab4 = st.tabs([" EDA", " Supervisionado", " Não Supervisionado", " Inferência"])

with tab1:
    st.subheader("Exploração de Dados (EDA)")
    st.write("Resumo estatístico das variáveis numéricas:")
    st.dataframe(df.describe().T, use_container_width=True)


    miss = df.isna().sum().sort_values(ascending=False)
    if miss.sum() > 0:
        st.write("Valores ausentes por coluna:")
        st.dataframe(miss[miss>0], use_container_width=True)
        impute = st.checkbox("Preencher valores ausentes automaticamente (média p/ numéricos)", value=True)
        if impute:
            for c in df.columns:
                if df[c].isna().any():
                    if pd.api.types.is_numeric_dtype(df[c]):
                        df[c] = df[c].fillna(df[c].mean())
                    else:
                        df[c] = df[c].fillna(df[c].mode().iloc[0])
            st.success("Valores ausentes preenchidos.")
    else:
        st.info("Sem valores ausentes detectados.")


    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != (target_col or "")]
    if numeric_cols:
        sel = st.multiselect("Escolha colunas numéricas para histogramas", numeric_cols, default=numeric_cols[:3])
        for c in sel:
            fig = px.histogram(df, x=c, nbins=30, title=f"Distribuição de {c}")
            st.plotly_chart(fig, use_container_width=True)

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=False, aspect="auto", title="Matriz de Correlação (numéricos)")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Aprendizagem Supervisionada")
    if not target_col:
        st.warning("Selecione uma coluna alvo na barra lateral para usar esta aba.")
    else:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        if task_type == "auto":
            if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
                task = "regression"
            else:
                task = "classification"
        else:
            task = task_type

        st.write(f"**Tarefa detectada:** `{task}` — Alvo: `{target_col}`")

        test_size = st.slider("Proporção de teste", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random State", min_value=0, max_value=9999, value=42, step=1)
        scale = st.checkbox("Padronizar features (StandardScaler)", value=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if task=="classification" and y.nunique()>1 else None)

        if task == "classification":
            model_name = st.selectbox("Modelo", ["LogisticRegression", "RandomForestClassifier", "SVC", "KNeighborsClassifier"])
            if model_name == "LogisticRegression":
                C = st.slider("C (inverso da regularização)", 0.01, 10.0, 1.0)
                clf = Pipeline([("scaler", StandardScaler())]) if scale else None
                model = LogisticRegression(max_iter=500, C=C, n_jobs=None)
                steps = []
                if scale: steps.append(("scaler", StandardScaler()))
                steps.append(("model", model))
                pipe = Pipeline(steps)
            elif model_name == "RandomForestClassifier":
                n_estimators = st.slider("Árvores (n_estimators)", 50, 400, 200, 50)
                max_depth = st.slider("Profundidade máxima", 2, 20, 8, 1)
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1)
                steps = [("model", model)]
                pipe = Pipeline(steps)
            elif model_name == "SVC":
                C = st.slider("C", 0.1, 10.0, 1.0)
                kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
                gamma = st.selectbox("Gamma", ["scale", "auto"])
                model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, random_state=random_state)
                steps = []
                if scale: steps.append(("scaler", StandardScaler()))
                steps.append(("model", model))
                pipe = Pipeline(steps)
            else:
                n_neighbors = st.slider("K (neighbors)", 1, 25, 5)
                model = KNeighborsClassifier(n_neighbors=n_neighbors)
                steps = []
                if scale: steps.append(("scaler", StandardScaler()))
                steps.append(("model", model))
                pipe = Pipeline(steps)

            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            proba = pipe.predict_proba(X_test)[:,1] if hasattr(pipe.named_steps["model"], "predict_proba") else None

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Acurácia", f"{accuracy_score(y_test, preds):.3f}")
            col2.metric("Precisão", f"{precision_score(y_test, preds, zero_division=0):.3f}")
            col3.metric("Recall", f"{recall_score(y_test, preds, zero_division=0):.3f}")
            col4.metric("F1-Score", f"{f1_score(y_test, preds, zero_division=0):.3f}")
            if proba is not None and y.nunique()==2:
                col5.metric("ROC AUC", f"{roc_auc_score(y_test, proba):.3f}")
            else:
                col5.metric("ROC AUC", "—")

           
            cm = confusion_matrix(y_test, preds)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title="Matriz de Confusão")
            fig_cm.update_xaxes(title_text="Predito")
            fig_cm.update_yaxes(title_text="Verdadeiro")
            st.plotly_chart(fig_cm, use_container_width=True)

            if proba is not None and y.nunique()==2:
                fpr, tpr, _ = roc_curve(y_test, proba)
                roc_auc = auc(fpr, tpr)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc:.3f})"))
                fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Aleatório", line=dict(dash="dash")))
                fig_roc.update_layout(title="Curva ROC", xaxis_title="Falso Positivo", yaxis_title="Verdadeiro Positivo")
                st.plotly_chart(fig_roc, use_container_width=True)

        else:  
            model_name = st.selectbox("Modelo", ["LinearRegression", "RandomForestRegressor", "SVR", "KNeighborsRegressor"])
            if model_name == "LinearRegression":
                model = LinearRegression()
                steps = [("model", model)]
                pipe = Pipeline(steps)
            elif model_name == "RandomForestRegressor":
                n_estimators = st.slider("Árvores (n_estimators)", 50, 400, 200, 50)
                max_depth = st.slider("Profundidade máxima", 2, 30, 10, 1)
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1)
                steps = [("model", model)]
                pipe = Pipeline(steps)
            elif model_name == "SVR":
                C = st.slider("C", 0.1, 10.0, 1.0)
                epsilon = st.slider("Epsilon", 0.01, 1.0, 0.1)
                kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
                model = SVR(C=C, epsilon=epsilon, kernel=kernel)
                steps = []
                if scale: steps.append(("scaler", StandardScaler()))
                steps.append(("model", model))
                pipe = Pipeline(steps)
            else:
                n_neighbors = st.slider("K (neighbors)", 1, 25, 5)
                model = KNeighborsRegressor(n_neighbors=n_neighbors)
                steps = []
                if scale: steps.append(("scaler", StandardScaler()))
                steps.append(("model", model))
                pipe = Pipeline(steps)

            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            mae = mean_absolute_error(y_test, preds)
            rmse = mean_squared_error(y_test, preds, squared=False)
            r2 = r2_score(y_test, preds)

            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{mae:.3f}")
            col2.metric("RMSE", f"{rmse:.3f}")
            col3.metric("R²", f"{r2:.3f}")

            
            fig_scatter = px.scatter(x=y_test, y=preds, labels={'x':'Verdadeiro', 'y':'Predito'}, title="Verdadeiro vs Predito")
            st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.subheader("Aprendizagem Não Supervisionada (K-Means + PCA)")
    features = df.drop(columns=[target_col]) if target_col else df.copy()
    # Garantir apenas numéricos
    features = features.select_dtypes(include=[np.number]).copy()
    n_components = st.slider("Componentes do PCA (para visualização)", 2, min(3, features.shape[1]) if features.shape[1]>=3 else 2, min(2, features.shape[1]))
    if features.shape[1] < 2:
        st.warning("São necessárias pelo menos 2 features numéricas para PCA e K-Means.")
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features.values)

        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        st.write("Variância explicada por componente:", np.round(pca.explained_variance_ratio_, 3))

        k = st.slider("Número de clusters (k)", 2, min(10, len(features)//2 if len(features)//2>=2 else 10), 3)
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, clusters)
        st.metric("Silhouette Score", f"{sil:.3f}")

        
        if n_components >= 2:
            pca_df = pd.DataFrame(X_pca[:, :2], columns=["PC1","PC2"])
            pca_df["cluster"] = clusters.astype(str)
            if target_col is not None and df[target_col].nunique() <= 10:
                pca_df["label"] = df[target_col].astype(str).values
                fig1 = px.scatter(pca_df, x="PC1", y="PC2", color="cluster", symbol="label", title="PCA 2D — Clusters vs Rótulos")
            else:
                fig1 = px.scatter(pca_df, x="PC1", y="PC2", color="cluster", title="PCA 2D — Clusters (K-Means)")
            st.plotly_chart(fig1, use_container_width=True)

        
        with st.expander("🔧 Curva do Cotovelo (Elbow)"):
            inertias = []
            ks = list(range(2, min(12, max(3, len(features)//2))+1))
            for kk in ks:
                km = KMeans(n_clusters=kk, n_init=10, random_state=42).fit(X_scaled)
                inertias.append(km.inertia_)
            fig_elbow = px.line(x=ks, y=inertias, markers=True, labels={"x":"k", "y":"Inércia"}, title="Curva do Cotovelo")
            st.plotly_chart(fig_elbow, use_container_width=True)

with tab4:
    st.subheader("Inferência / Predição com modelo treinado (rápido)")
    st.write("Use a aba **Supervisionado** para treinar um modelo. Depois, utilize os controles abaixo para fazer uma predição rápida com os mesmos hiperparâmetros.")

    if target_col:
        X_all = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).copy()
    else:
        X_all = df.select_dtypes(include=[np.number]).copy()

    if X_all.empty:
        st.warning("Nenhuma feature numérica disponível para inferência.")
    else:
        st.write("Ajuste os valores das features:")
        user_input = {}
        for col in X_all.columns[:30]:  
            val = float(X_all[col].median())
            user_input[col] = st.number_input(col, value=val)
        X_user = pd.DataFrame([user_input])

        
        model_kind = st.selectbox("Tipo de modelo para inferência", ["classification", "regression"] if (task_type=="auto" and target_col) else [task_type if target_col else "classification"])
        scale_inf = st.checkbox("Padronizar features (StandardScaler)", value=True, key="scale_inf")
        if model_kind == "classification":
            model_sel = st.selectbox("Modelo", ["LogisticRegression", "RandomForestClassifier", "SVC", "KNeighborsClassifier"], key="m_cls")
            if model_sel == "LogisticRegression":
                C = st.slider("C", 0.01, 10.0, 1.0, key="C_cls")
                steps = []
                if scale_inf: steps.append(("scaler", StandardScaler()))
                steps.append(("model", LogisticRegression(max_iter=500, C=C)))
            elif model_sel == "RandomForestClassifier":
                n_estimators = st.slider("Árvores", 50, 400, 200, 50, key="ne_cls")
                max_depth = st.slider("Profundidade", 2, 20, 8, 1, key="md_cls")
                steps = [("model", RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42))]
            elif model_sel == "SVC":
                C = st.slider("C", 0.1, 10.0, 1.0, key="C_svc")
                kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], key="k_svc")
                gamma = st.selectbox("Gamma", ["scale", "auto"], key="g_svc")
                steps = []
                if scale_inf: steps.append(("scaler", StandardScaler()))
                steps.append(("model", SVC(C=C, kernel=kernel, gamma=gamma, probability=True, random_state=42)))
            else:
                n_neighbors = st.slider("K", 1, 25, 5, key="k_knn")
                steps = []
                if scale_inf: steps.append(("scaler", StandardScaler()))
                steps.append(("model", KNeighborsClassifier(n_neighbors=n_neighbors)))
            model_pipe = Pipeline(steps)
           
            if target_col:
                model_pipe.fit(X_all, df[target_col])
                pred = model_pipe.predict(X_user)[0]
                proba = model_pipe.predict_proba(X_user)[0,1] if hasattr(model_pipe.named_steps["model"], "predict_proba") else None
                st.success(f"Predição da classe: **{pred}**" + (f" | Prob. classe 1: **{proba:.3f}**" if proba is not None else ""))
            else:
                st.info("Sem alvo disponível; o modelo será treinado apenas para fins ilustrativos (sem rótulos).")
        else:
            model_sel = st.selectbox("Modelo", ["LinearRegression", "RandomForestRegressor", "SVR", "KNeighborsRegressor"], key="m_reg")
            if model_sel == "LinearRegression":
                steps = [("model", LinearRegression())]
            elif model_sel == "RandomForestRegressor":
                n_estimators = st.slider("Árvores", 50, 400, 200, 50, key="ne_reg")
                max_depth = st.slider("Profundidade", 2, 30, 10, 1, key="md_reg")
                steps = [("model", RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42))]
            elif model_sel == "SVR":
                C = st.slider("C", 0.1, 10.0, 1.0, key="C_svr")
                epsilon = st.slider("Epsilon", 0.01, 1.0, 0.1, key="e_svr")
                kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], key="k_svr")
                steps = []
                if scale_inf: steps.append(("scaler", StandardScaler()))
                steps.append(("model", SVR(C=C, epsilon=epsilon, kernel=kernel)))
            else:
                n_neighbors = st.slider("K", 1, 25, 5, key="k_knnr")
                steps = []
                if scale_inf: steps.append(("scaler", StandardScaler()))
                steps.append(("model", KNeighborsRegressor(n_neighbors=n_neighbors)))
            model_pipe = Pipeline(steps)
            if target_col and pd.api.types.is_numeric_dtype(df[target_col]):
                model_pipe.fit(X_all, df[target_col])
                pred = model_pipe.predict(X_user)[0]
                st.success(f"Predição: **{pred:.3f}**")
            else:
                st.info("Para inferência de regressão, selecione um dataset com alvo numérico.")
