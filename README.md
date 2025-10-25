#  Machine Learning Aplicado à Saúde — Streamlit App

Este projeto foi desenvolvido como **atividade final da disciplina de Machine Learning Aplicado à Saúde**, com o objetivo de demonstrar o uso prático de técnicas de **aprendizado supervisionado e não supervisionado** utilizando o framework **Streamlit**.

A aplicação permite analisar dados de saúde, treinar modelos preditivos e explorar agrupamentos de pacientes por similaridade, tornando o processo interativo e acessível.

---

##  Objetivo do Projeto

Aplicar técnicas de **Machine Learning** em um contexto real da área da saúde, abordando todo o ciclo:
1. Carregamento e tratamento dos dados.  
2. Análise exploratória (EDA).  
3. Modelagem supervisionada (regressão).  
4. Modelagem não supervisionada (clusterização).  
5. Interpretação dos resultados em uma interface interativa.

O dataset principal utilizado é o **Diabetes** disponível no `scikit-learn`, mas o usuário também pode enviar seu próprio arquivo CSV para análise.

---

##  Funcionalidades Principais

### Análise Exploratória (EDA)
- Visualização PCA (Projeção em 2D) para entender padrões nos dados.  
- Matriz de correlação entre variáveis.  
- Estatísticas básicas do dataset.

###  Aprendizado Supervisionado (Regressão)
Modelos disponíveis:
- **LinearRegression**
- **Ridge Regression**
- **Lasso Regression**
- **RandomForestRegressor**

Métricas exibidas:
- MAE (Erro Absoluto Médio)
- RMSE (Raiz do Erro Quadrático Médio)
- R² (Coeficiente de Determinação)
- Cross-Validation média e desvio-padrão  

Gráficos interativos:
- Verdadeiro vs. Predito  
- Distribuição dos resíduos  
- Importância ou coeficiente dos atributos  

###  Aprendizado Não Supervisionado (Clusterização)
- Algoritmo **K-Means**
- Redução de dimensionalidade com **PCA (2D)**  
- Cálculo do **Silhouette Score**
- Perfil médio de cada cluster (médias dos atributos)

---

##  Estrutura do Projeto

