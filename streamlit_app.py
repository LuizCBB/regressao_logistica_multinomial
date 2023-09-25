# Imports
import time
import numpy as np
import pandas as pd
import streamlit as st
import sklearn.metrics
import sklearn.datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from PIL import Image

# CSS
st.markdown(""" <style>

#div_algorithm_name {
  background-color: lightblue;
  text-align: left;
  height: 40px;
  max-width: 2000px; 
  border-radius: 10px;
  padding-left: 10px;
}
 
#algorithm_name_title{
    font-size:24px;
    font-family: 'Arial Narrow';
    color: black;
}

</style> """, unsafe_allow_html=True)

# Funções Python

def cria_modelo(parameters, dados, split):
    
    # Extrai os dados de treino e teste
    #X_treino, X_teste, y_treino, y_teste = prepara_dados(Data, Split) 
    X_treino, X_teste, y_treino, y_teste = train_test_split(dados.data, dados.target, test_size = float(split), random_state = 42)

    # Cria o modelo
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    clf = LogisticRegression(penalty = parameters['Penality'], 
                             solver = parameters['Solver'], 
                             max_iter = int(parameters['Max_Iteration']), 
                             tol = float(parameters['Tol']))

    # Treina o modelo
    clf = clf.fit(X_treino, y_treino)

    # Faz previsões
    prediction = clf.predict(X_teste)
    
    # Calcula a acurácia
    accuracy = sklearn.metrics.accuracy_score(y_teste, prediction)

    # Calcula a confusion matrix
    cm = confusion_matrix(y_teste, prediction)

    # Dicionário com os resultados
    dict_value = {"modelo":clf, "acuracia": accuracy, "previsao":prediction, "y_real": y_teste, "Metricas":cm, "X_teste": X_teste }
       
    return(dict_value)


#Título do algoritmo
st.markdown('<div id="div_algorithm_name"> <p id = "algorithm_name_title"> Regressão Logística Multivariada</p> </div></br>', unsafe_allow_html=True)


##### Programando o Corpo da Aplicação Web ##### 

dataset_selectbox = st.selectbox('1 - Selecione um banco de dados',('Iris', 'Wine'))

if dataset_selectbox == 'Iris':
    dataset = sklearn.datasets.load_iris()
    st.markdown("""<p style='text-align: justify; font-size: 15px;'>
    O <i>dataset</i> Iris é um conjunto de dados multivariados que consiste de amostras de cada uma de três espécies de plantas
    do gênero Iris (<i>Iris setosa</i>, <i>Iris virginica</i> e <i>Iris versicolor</i>). Quatro variáveis foram medidas em cada amostra: 
    comprimento da sépala (sepal length), largura da sépala (sepal width), comprimento da pétala (petal length) e a largura da pétala (petal width).
    Todas as medidas estão em centímetros. Veja mais informações sobre este <i>dataset</i> <a href='https://archive.ics.uci.edu/dataset/53/iris'>aqui</a>.</p>
    """, unsafe_allow_html=True)    
    image = Image.open('flores_de_Iris.png')
    st.image(image, caption='Diferenças entre flores de Iris. Fonte: Wikipédia (https://pt.wikipedia.org/wiki/Conjunto_de_dados_flor_Iris)')
        
elif dataset_selectbox == 'Wine':
    dataset = sklearn.datasets.load_wine()
    st.markdown("""<p style='text-align: justify; font-size: 15px;'>
    O <i>dataset</i> Wine é resultado de uma análise química de vinhos cuja matéria prima é proveniente da mesma região da Itália, mas derivada de três cultivares
    de videiras diferentes. Dessa forma, tem-se três categorias (identificadas no <i>dataset</i> como class_0, class_1 e class_2) para a variável a ser predita (variável
    <i>target</i>). A análise determinou as quantidades de 13 constituintes (que representam as variávels preditoras ou explicativas) encontrados em cada um dos três tipos
    de vinhos: álcool (alcohol), ácido málico (malic_acid), cinzas (ash), alcalinidade das cinzas (alcalinity_of_ash), magnésio (magnesium),
    fenóis totais (total_phenols), flavonóides (flavanoids), fenóis não flavonóides (nonflavanoid_phenols), proantocianinas (proanthocyanins),
    intensidade de cor (color_intensity), matiz (hue), OD280/OD315 de vinhos diluídos (OD280/OD315_of_diluted_wines) e prolina (proline). Veja mais 
    informações sobre este <i>dataset</i> <a href='https://archive.ics.uci.edu/dataset/109/wine'>aqui</a>.</p>
    """, unsafe_allow_html=True)


# Extrai a variável alvo
targets = dataset.target_names

# Prepara o dataframe com os dados
dataframe = pd.DataFrame (dataset.data, columns = dataset.feature_names)
dataframe['target'] = pd.Series(dataset.target)
dataframe['target labels'] = pd.Series(targets[i] for i in dataset.target)

# Mostra o dataset selecionado pelo usuário
st.markdown("<p style='text-align: justify; font-size: 15px;'>2 - Visualizando o <i>dataset</i></p>", unsafe_allow_html = True)
st.write(dataframe)

st.markdown("<p style='text-align: justify; font-size: 15px;'>3 - Construindo o modelo de classificação utilizando regressão logística multivariada</p>", unsafe_allow_html = True)
Split = st.slider('Escolha o percentual dos dados que ficará para teste (padrão = 30%):', 10, 90, 30)
Split = Split/100
st.markdown("<p style='text-align: justify; font-size: 14px;'>Selecione os hiperparâmetros para o modelo de regressão logística multivariada:<p>", unsafe_allow_html = True)
Solver = st.selectbox('Selecione o solver (padrão = lbfgs):', ('lbfgs', 'newton-cg', 'liblinear', 'sag'))
Penality = st.radio("Selecione uma regularização (padrão = none):", ('none', 'l2'))  
Tol = st.text_input("Selecione a tolerância para critério de parada (padrão = 1e-4):", "1e-4")
Max_Iteration = st.text_input("Número de Iterações (padrão = 50):", "50") 

# Dicionário Para os Hiperparâmetros
parameters = {'Penality': Penality, 'Tol': Tol, 'Max_Iteration': Max_Iteration, 'Solver': Solver}

st.markdown("<p style='text-align: justify; font-size: 15px;'>4 - Treinando, testando e gerando métricas de avaliação do o modelo</p>", unsafe_allow_html = True)
if(st.button("Clique para treinar, testar e gerar as métricas do modelo")):
    
    # Info para o usuário
    st.write('Carregando os dados e treinando o modelo. Por favor seja paciente ...')
 
    # Cria e treina o modelo
    modelo = cria_modelo(parameters, dataset, Split) 
    
    # Barra de progressão
    my_bar = st.progress(0)

    # Mostra a barra de progressão com percentual de conclusão
    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)

    # Info de sucesso
    st.success("Modelo treinado!")
    
    
    # Extrai os labels reais
    labels_reais = [targets[i] for i in modelo["y_real"]]

    # Extrai os labels previstos
    labels_previstos = [targets[i] for i in modelo["previsao"]]

    st.markdown("<p style='text-align: justify; font-size: 15px;'>Previsões do modelo nos dados de teste (" + str(int(Split*100)) + "% do total de dados):</p>", unsafe_allow_html = True)

    # Mostra o resultado
    st.write(pd.DataFrame({"Valor Real" : modelo["y_real"], 
                           "Label Real" : labels_reais, 
                           "Valor Previsto" : modelo["previsao"], 
                           "Label Previsto" :  labels_previstos,})) 
    
    st.markdown("<p style='text-align: justify; font-size: 15px;'>Métricas:</p>", unsafe_allow_html = True)
    # Extrai as métricas
    matriz = modelo["Metricas"]

    st.markdown("<p style='text-align: justify; font-size: 15px;'>Matriz de confusão nos dados de teste</p>", unsafe_allow_html = True)

    # Mostra a matriz de confusão
    st.write(matriz)

    # Mostra a acurácia
    st.markdown("<p style='text-align: justify; font-size: 15px;'>Acurácia do modelo nos dados de teste:  " + str(round(modelo["acuracia"], 2)*100) + "%</p>", unsafe_allow_html = True)
    

    
    
 









                            
            
        












                            
            
        


