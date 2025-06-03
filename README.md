# APS - Natural Language Processing

## Análise de Desempenho de Modelos com base na Quantidade de Dados Disponíveis

### Objetivo
O objetivo deste projeto consiste em analisar o desempenho de diferentes abordagens relacionadas ao processamento de linguagem natural (NLP) com base na quantidade de dados disponíveis. Os modelos escolhidos para análise são:

- **Bag of Words (BoW - TF-IDF) + Regressão Logística**: abordagem que consiste em transformar o texto em uma representação numérica, e em seguida, aplicar um modelo de regressão logística para classificação. O vetorizador utilizado é o `TfidfVectorizer` do `sklearn`, que transforma o texto em uma matriz de termos ponderados por frequência inversa de documento (TF-IDF). Desta forma, é possível capturar a importância relativa de cada termo no contexto do documento.
- **BERT + Regressão Logística**: abordagem que utiliza o modelo BERT para gerar embeddings dos textos, seguido por uma regressão logística para classificação. O modelo BERT, diferentemente do BoW, é capaz de capturar o contexto e a semântica das palavras, proporcionando uma representação mais rica e informativa dos textos.

### Metodologia
A metodologia adotada para a análise de desempenho dos modelos inclui as seguintes etapas:

1. **Coleta de Dados**: para a análise, escolhemos o dataset [Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) do Kaggle, que contém 52.681, com 3 colunas:
    - `unique_id`: identificador único da linha
    - `statement`: texto a ser classificado
    - `status`: rótulo de classificação, que pode ser:
        - `Anxiety`
        - `Depression`
        - `Normal`
        - `Suicidal`
        - `Bipolar`

    Desta forma, o dataset é um problema de classificação multiclasse, onde o objetivo é classificar os textos em uma das cinco categorias mencionadas.

2. **Pré-processamento dos Dados**: nesta etapa, foi realizada a remoção de linhas nulas, de forma a garantir que o dataset esteja completo e pronto para análise.

3. **Divisão dos Dados**: para a realização do estudo dos desempenhos dos modelos, dividimos a análise em diferentes conjuntos de dados, variando a quantidade destes da seguinte forma:
    - 1.000 linhas
    - 5.000 linhas
    - 10.000 linhas
    - 20.000 linhas
    - Dataset completo (~ 53.000 linhas)

4. **Treinamento e Avaliação dos Modelos**: para cada conjunto de dados, os modelos foram treinados e avaliados utilizando de acordo com as seguintes métricas:
    - **Acurácia**: proporção de previsões corretas em relação ao total de previsões.
    - **F1-Score**: média harmônica entre as métricas Precision e Recall:
      
        - **Precision**: proporção de previsões positivas corretas em relação ao total de previsões positivas.
  
          $\text{Precision} \=\ \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}}$, onde:
  
          - TP (True Positives) são os verdadeiros positivos
          - FP (False Positives) são os falsos positivos
    
        - **Recall**: proporção de previsões positivas corretas em relação ao total de casos positivos reais.
        
          $\text{Recall} \=\ \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}}$, onde:
          
          - TP (True Positives) são os verdadeiros positivos
          - FN (False Negatives) são os falsos negativos

      
Além disso, foram gerados gráficos de matriz de confusão para cada modelo, permitindo uma visualização clara do desempenho do modelo em cada classe.

5. **Análise dos Resultados**: os resultados foram analisados para identificar tendências e padrões de desempenho dos modelos em relação à quantidade de dados disponíveis.

