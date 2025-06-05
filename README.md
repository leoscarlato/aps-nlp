# Análise de Desempenho de Modelos com base na Quantidade de Dados Disponíveis

Membros: Leonardo Scarlato e João Alfredo Cardoso Lamy

## Introdução
No cenário atual de tecnologias de NLP, observam-se discussões sobre diferentes metodologias e ferramentas utilizadas em projetos e sistemas de classificação. De um lado, estão as abordagens tradicionais como Bag of Words (BoW), combinadas com algoritmos lineares como Regressão Logística, que se destacam pela simplicidade e desempenho competitivo em cenários com dados limitados. Do outro, os modelos baseados em transformers, como o BERT, vêm ganhando espaço por sua capacidade de capturar contexto, semântica e dependências complexas entre palavras, fatores que tornam essas soluções mais robustas, especialmente em contextos com maior volume de dados.

Este contraste, portanto, pode ser analisado de forma a comparar **como as diferentes abordagens desempenham em função de diferentes quantidades de dados**, buscando identificar em que ponto modelos mais simples deixam de ser suficientes e quando modelos mais complexos se tornam vantajosos. Para este estudo, foi utilizado como base o artigo [When BERT Meets Bilbo: A Learning Curve Analysis of Pretrained Language Models on Disease Classification](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-022-01829-2), onde os autores conduziram um estudo aplicado com dados referentes à área da saúde, comparando as curvas de aprendizado de diferentes modelos, incluíndo BoW e BERT, conforme o tamanho do conjunto de treinamento cresce.

## Objetivo
O objetivo deste projeto consiste em analisar o desempenho de diferentes abordagens relacionadas ao processamento de linguagem natural (NLP) com base na quantidade de dados disponíveis. Os modelos escolhidos para análise são:

- **Bag of Words (BoW - TF-IDF) + Regressão Logística**: abordagem que consiste em transformar o texto em uma representação numérica, e em seguida, aplicar um modelo de regressão logística para classificação. O vetorizador utilizado é o `TfidfVectorizer` do `sklearn`, que transforma o texto em uma matriz de termos ponderados por frequência inversa de documento (TF-IDF). Desta forma, é possível capturar a importância relativa de cada termo no contexto do documento.
- **BERT + Regressão Logística**: abordagem que utiliza o modelo BERT para gerar embeddings dos textos, seguido por uma regressão logística para classificação. O modelo BERT, diferentemente do BoW, é capaz de capturar o contexto e a semântica das palavras, proporcionando uma representação mais rica e informativa dos textos.

## Metodologia
A metodologia adotada para a análise de desempenho dos modelos inclui as seguintes etapas:

1. **Coleta de Dados**: para a análise, escolhemos o dataset [Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) do Kaggle, que contém 52.681 linhas e 3 colunas:
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
    - Dataset completo

4. **Treinamento e Avaliação dos Modelos**: para cada conjunto de dados, os modelos foram treinados e avaliados utilizando de acordo com as seguintes métricas:
    - **Acurácia**: proporção de previsões corretas em relação ao total de previsões.
    - **F1-Score**: média harmônica entre as métricas Precision e Recall:
  
        $\mathrm{F1} \=\ 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
            
        - **Precision**: proporção de previsões positivas corretas em relação ao total de previsões positivas.
  
          $\text{Precision} \=\ \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}}$, onde:
  
          - TP (True Positives) são os verdadeiros positivos
          - FP (False Positives) são os falsos positivos
    
        - **Recall**: proporção de previsões positivas corretas em relação ao total de casos positivos reais.
        
          $\text{Recall} \=\ \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}}$, onde:
          
          - TP (True Positives) são os verdadeiros positivos
          - FN (False Negatives) são os falsos negativos

      
Além disso, foram gerados gráficos de matriz de confusão para cada modelo, permitindo uma visualização clara do desempenho do modelo em cada classe.

## Principais resultados obtidos

Após o treinamento e aplicação dos modelos em diferentes quantidades de dados, obtivemos os seguintes resultados referentes às métricas pré-definidas:

![image](https://github.com/user-attachments/assets/2620bfc3-8a96-404a-8f34-16b8be68a630)

![image](https://github.com/user-attachments/assets/94423546-7aae-4c95-b462-4e8938899adb)

Com poucos dados (neste caso 1000 samples), observa-se que o BoW + Regressão Logística tende a apresentar maior acurácia geral, indicando que a representação captura os padrões mais simples quando o volume é limitado. No entanto, o BERT, mesmo com menos acertos brutos, mostra um F1‐score mais equilibrado, sinalizando que sua capacidade contextual ajuda a não negligenciar classes menos frequentes, mantendo uma harmonia melhor entre precisão e recall.

À medida que o conjunto de treinamento cresce (em algumas milhares de amostras), o BERT passa a superar o BoW tanto em acurácia quanto em F1. Com cerca de 20000 a 5000 samples, o poder de representação semântica do BERT se destaca e torna-se claramente vantajoso, obtendo ganhos mais expressivos em todas as classes.

Por fim, em volumes maiores de dados, a diferença entre os dois modelos se dilui.

![image](https://github.com/user-attachments/assets/69de77cb-42c6-4705-b7a3-1be62d612f66)

![image](https://github.com/user-attachments/assets/a3619dd9-df96-4f25-81d1-dec36877f5ed)

Além das duas métricas exibidas anteriormente, também é possível visualizar as curvas de aprendizado de cada modelo, para cada quantidade de dados.

Para o modelo BoW, é possível observar que, com poucos exemplos, ele ainda não domina todos os padrões textuais, resultando em desempenho relativamente baixo e maior diferença entre acurácia de treino e teste (indicando underfitting). À medida que aumentando o número de documentos, a curva de treino se torna estável e a curva de teste sobe de forma constante, mostrando melhor generalização. Com volumes intermediários de dados, essas curvas se aproximam, revelando que o BoW já “entende” boa parte do vocabulário e reduz o gap entre treino e teste. Por fim, em grandes quantidades de exemplos, ambas as curvas se estabilizam, sugerindo que o modelo atingiu seu limite de aprendizado.

Para o BERT, com poucos exemplos as curvas de treino e teste apresentam um comportamento oposto ao do BoW: a acurácia de treino fica muito alta desde o início, enquanto a de teste permanece baixa, indicando que o modelo está se ajustando demais aos poucos dados (overfitting). Ao fornecer mais documentos, o BERT passa a reduzir esse overfitting, isto é, a curva de treino começa a cair gradualmente, e a curva de teste sobe de forma contínua, demonstrando que o modelo começa a generalizar melhor. Em volumes intermediários, nota-se que o BERT alcança rapidamente uma diferença menor entre treino e teste e assume desempenho superior ao BoW, aproveitando sua representação contextual para extrair ganhos com cada novo lote de dados. Em grandes quantidades de exemplos, ambas as curvas do BERT se aproximam e se estabilizam, sugerindo que o modelo atingiu seu limite de aprendizado.

## Conclusão

Os resultados aqui obtidos seguem o padrão visto em [When BERT Meets Bilbo: A Learning Curve Analysis of Pretrained Language Models on Disease Classification](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-022-01829-2). Assim como neste estudo, observamos que:

- Em baixas quantidades de dados, Bag-of-Words + Regressão Logística apresenta maior acurácia, mas o BERT já equilibra melhor precisão e recall (maior F1);

- Em volumes intermediários (algumas milhares de amostras), o BERT supera de forma consistente o BoW em acurácia e F1, confirmando que seu poder semântico se destaca quando há dados suficientes;

- Em grandes volumes de dados, ambos convergem para desempenhos próximos.

Dessa forma, concluímos que a escolha entre modelos tradicionais e modelos baseados em transformers deve considerar não apenas a complexidade do problema, mas também a quantidade de dados disponível. O BoW se mostra eficiente e competitivo em cenários com dados limitados, enquanto o BERT demonstra sua superioridade em situações onde há dados suficientes para explorar sua capacidade contextual. No entanto, quando o volume de dados é alto, o custo computacional do BERT pode não justificar a pequena diferença de desempenho em relação ao BoW, tornando este último uma alternativa viável e eficiente. Assim, os resultados obtidos não apenas validam as tendências apontadas no artigo de referência, como também reforçam a importância de alinhar a escolha do modelo ao contexto e aos recursos do projeto.

## Vídeo explicativo do projeto
https://drive.google.com/file/d/10xuRy3nWLJBbg-FJudDExy0QUBo_e7C8/view?usp=drive_link
