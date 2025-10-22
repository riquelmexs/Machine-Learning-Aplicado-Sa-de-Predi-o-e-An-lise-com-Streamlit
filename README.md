# 🩺 Machine Learning Aplicado à Saúde: Predição e Análise com Streamlit

Este projeto apresenta uma aplicação web interativa desenvolvida em **Streamlit** que demonstra o uso de **técnicas de Machine Learning supervisionadas e não supervisionadas** aplicadas à área da saúde.  
A solução tem como objetivo mostrar, de forma prática e visual, como modelos de aprendizado de máquina podem **analisar, prever e agrupar dados médicos**, auxiliando em diagnósticos e tomadas de decisão.

---

## 🎯 Objetivo do Projeto

O objetivo principal é desenvolver uma aplicação capaz de:
- Realizar **análise exploratória de dados (EDA)** em datasets de saúde;  
- Treinar e avaliar **modelos supervisionados** (classificação e regressão);  
- Aplicar **métodos não supervisionados** (K-Means e PCA) para agrupar dados;  
- Permitir **interação com o usuário** para gerar novas previsões;  
- Exibir resultados de forma visual e acessível.

---

## ⚙️ Tecnologias Utilizadas

- **Python 3.11+**  
- **Streamlit** — interface web interativa  
- **Scikit-learn (sklearn)** — modelos de Machine Learning  
- **Pandas e NumPy** — manipulação e análise de dados  
- **Plotly** — visualizações interativas  

---

## 🧠 Datasets Utilizados

A aplicação permite o uso de **datasets embutidos** do `sklearn` ou arquivos **CSV personalizados**.  
Os datasets principais são:

- 🩸 **Câncer de Mama (Breast Cancer Dataset)** — classificação binária (diagnóstico).  
- 💉 **Diabetes Dataset** — regressão (progressão da doença).

Também é possível enviar um **CSV próprio** contendo dados médicos ou laboratoriais, definindo manualmente a coluna alvo.

---

## 🔍 Estrutura da Aplicação

A interface do Streamlit é dividida em **quatro abas principais**:

### 1. 🔎 EDA (Exploração de Dados)
- Mostra informações gerais do dataset: número de amostras, variáveis e tipo de tarefa.  
- Exibe estatísticas descritivas, tratamento de valores ausentes e gráficos de distribuição.  
- Gera uma **matriz de correlação** interativa para identificar relações entre variáveis.

### 2. 🧠 Aprendizagem Supervisionada
- O usuário escolhe o tipo de modelo e ajusta hiperparâmetros.  
- Modelos disponíveis:
  - *Classificação*: Logistic Regression, Random Forest, SVC, KNN  
  - *Regressão*: Linear Regression, Random Forest Regressor, SVR, KNN Regressor  
- Mostra métricas de desempenho: **Acurácia, Precisão, Recall, F1-Score, ROC AUC**, **MAE**, **RMSE** e **R²**.  
- Exibe gráficos como **Matriz de Confusão** e **Curva ROC**.

### 3. 🧩 Aprendizagem Não Supervisionada
- Executa **K-Means** para agrupamento de dados e **PCA** para redução de dimensionalidade.  
- Gera gráficos 2D dos clusters e calcula o **Silhouette Score**.  
- Exibe também a **Curva do Cotovelo (Elbow)** para auxiliar na escolha do número ideal de clusters.

### 4. 🧪 Inferência (Predição)
- O usuário insere valores manualmente nas features.  
- O modelo treinado realiza uma **predição em tempo real**, retornando o resultado (classe ou valor previsto).

---

## 🖥️ Como Executar o Projeto

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/seu-usuario/ml_saude_streamlit.git
   cd ml_saude_streamlit
   ```

2. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Execute o aplicativo:**
   ```bash
   streamlit run app.py
   ```

4. Acesse em [http://localhost:8501](http://localhost:8501)

---

## ☁️ Deploy no Streamlit Cloud

1. Crie um repositório no GitHub com os arquivos:
   - `app.py`  
   - `requirements.txt`  
   - `README.md`  

2. Acesse [https://streamlit.io/cloud](https://streamlit.io/cloud)  
3. Conecte sua conta do GitHub e selecione o repositório.  
4. Escolha o arquivo `app.py` como principal e publique.  

O link final do aplicativo ficará no formato:  
**https://ml-saude-seunome.streamlit.app**

---

## 📂 Estrutura de Pastas
```
ml_saude_streamlit/
├── app.py
├── requirements.txt
└── README.md
```

---

## 👨‍💻 Autor

**Pablo Riquelme**  
Curso: **Análise e Desenvolvimento de Sistemas (ADS)**  
Faculdade Senac Recife
