#!/usr/bin/env python
# coding: utf-8

# In[1]:


# LISTAS
lista1 = [1, 2, 3] # O Python só identifica como uma lista através dos colchetes.


# In[2]:


type(lista1)


# In[3]:


lista2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


# In[4]:


type(lista2)


# In[5]:


lista1[0] # Para identificar um elemento (primeiro) da lista 1.


# In[6]:


lista1[2]


# In[7]:


lista2[0]


# In[8]:


lista2[2]


# In[9]:


lista2[0][2] # Seleciona o terceiro elemento da primeira lista.


# In[11]:


lista2[2][2]


# In[12]:


select = lista2[1][2]


# In[13]:


print(select)


# In[14]:


type(select)


# In[18]:


select = lista2[2]


# In[19]:


print(select)
type(select)


# In[27]:


import random
cidades = ["São Paulo", "Rio de Janeiro", "Belo Horizonte", "Natal"]
escolhida = random.choice(cidades)
print("A cidade escolhida é:", escolhida)


# In[28]:


a = [1, 2, 3]


# In[29]:


a.append(15) # Comando para adicionar um novo elemento à lista.


# In[30]:


print(a)


# In[31]:


b = [7, 8, 9]


# In[32]:


for item in b: # Para cada um dos elementos de b...
    a.append(item) # Inclui cada elemento de b dentro da lista a.


# In[33]:


print(a)


# In[34]:


# TRANFORMANDO LISTAS FLOAT EM INT
num = 8
type(num)


# In[35]:


float(num) # Converte a variável num de inteiro para float.


# In[36]:


num1 = 3.5
type(num1)


# In[37]:


int(num1) # Converte a variável num1 de float para inteiro.


# In[38]:


x = [2, 4, 10, 6]
type(x)


# In[41]:


print(x)


# In[44]:


nova = []
for item in x: 
    nova.append(float(item))


# In[45]:


print(nova)


# In[1]:


# TUPLAS

### Ao contrário das listas, as tuplas são colocadas em parênteses.
### As tuplas são listas que não podem ser alteradas.

lista = [3, 6, 4]
lista[0] = 20 # Na lista é possível substituir os valores.
print(lista)


# In[2]:


del lista[2] # Deleta o item dois da lista.
print(lista)


# In[3]:


tupla = (5, 3, 8)
tupla[0] = 30
print(tupla) # Retorna uma mensagem de erro informando que o intem não pode ser atribuído.


# In[4]:


del tupla[2]
print(tupla) # Retorna mensagem de erro que não é permitido deletar item.


# In[5]:


tupla.append(10)
print(tupla) # Retorna mensagem de que não é permitido adicionar valores à tupla.


# In[7]:


nova_tupla = tuple(lista) # Transforma a lista em tupla.
print(nova_tupla)


# In[8]:


type(nova_tupla)


# In[9]:


numero = (3)
print(numero)
type(numero) # Informa que é um inteiro, pois para ser considerado tupla
## deve estar em forma de lista com vígulas entre os itens.


# In[10]:


# DICIONÁRIOS

### Dicionários são apresentados com notação de chaves.
### Exemplo: dicionario = {chave:valor}.

dicionario = {'Curso' : 'Python para Machine Learning',
             'Produtor' : 'Didática Tech',
             'Preço' : 'Gratuito',
             'Nota' : 10} # Também podem ser atruídos valores.
dicionario['Preço']


# In[12]:


dicionario['Nota']


# In[13]:


a = dicionario['Produtor'] # Nova variável a recebe o atributo 'Pordutor' do dicionário.
print(a)


# In[14]:


dicionario['Preço'] = 'R$ 100,00'
dicionario['Preço']


# In[15]:


print(dicionario)


# In[17]:


dicionario['Pré-requisito'] = 'Python básico' # Para adicionar uma nova chave.
print(dicionario)


# In[18]:


dicionario.keys()


# In[19]:


dicionario.values() # Puxa os valores do dicionário


# In[23]:


dicionario.items() # Puxa todas as informações do dicionário.


# In[24]:


dicionario.clear() # Apaga todas as informações do dicionário.


# In[25]:


print(dicionario)


# In[29]:


# MANIPULANDO STRINGS

frase = 'Estou gostando do curso.'
frase[2:6]


# In[30]:


frase[2]


# In[31]:


frase[9:12]


# In[33]:


frase[19:24]


# In[34]:


frase[0:] # Imprime do primeiro caracter até o último.


# In[35]:


frase[2:13:2] # Puxa da posição 2 até a 13 pulando de 2 em 2.


# In[37]:


frase.count('t') # Conta a quantidade de letras 't'.


# In[39]:


frase.count(' ') # Conta os espaços.


# In[40]:


len(frase) # Conta o número de caracteres.


# In[41]:


frase.replace('s', 'x')


# In[42]:


frase.replace('curso', 'aprendizado')


# In[1]:


# FUNÇÃO LAMBDA

### Relembrando os conceitos da criação de funções.

def somaQuadrados(a, b): # Define o nome da função e os parâmetros utilizados.
    somaQ = a**2 + b**2 # Informa a fórmula da função.
    return somaQ # Pede o resultado da função.


# In[2]:


somaQuadrados(4, 7)


# In[3]:


### A função lambda cria novas funções, mas de forma resumida.

somaQuadrados2 = lambda a, b: a**2 + b**2


# In[4]:


somaQuadrados2(5, 2)


# In[5]:


calculadoraQuadrado = lambda a: a**2


# In[6]:


calculadoraQuadrado(4)


# In[7]:


x = lambda f: f/2


# In[8]:


x(9)


# In[36]:


# FUNÇÃO MAP

### Da mesma forma do lambda, o map resume o código.

kmh = [40, 50, 56, 64,  73, 79, 85, 96, 100, 120] # Lista de velocidades kilometros por hora.

mph = [] # Criando lista vazia.
for i in kmh: # Para cada item da lista kph...
    mph.append(i/1.61) # Adicione os valores na lista vazia divididos por 1.61.
print(mph)


# In[37]:


myRoundedList = [round(i,2) for i in mph] 


# In[28]:


print(myRoundedList)


# In[40]:


mph2 = list(map(lambda x : x/1.61, kmh)) # Deve ser transformado em uma lista quando usa a função map().
print(mph2)


# In[41]:


myRoundedList2 = [round(i,2) for i in mph2] 
print(myRoundedList2)


# In[42]:


# LIST COMPREHENSION

### Permite criar funções ainda mais reduzidas.

mph3 = [x/1.61 for x in kmh]
print(mph3)


# In[43]:


myRoundedList3 = [round(i,2) for i in mph3] 
print(myRoundedList3)


# In[44]:


caracteres = [i for i in "Didática Tech"] # Puxa o caractere i para todos os elementos na string "Didática Tech".
print(caracteres)


# In[1]:


# Pacote Numpy

import numpy
a = numpy.array([1, 2, 3]) # Array de uma dimensão.
print(a)


# In[4]:


import numpy as np # Importa o pacote como np para facilitar na escrita dos códigos
b = np.array([1, 4, 7, 3, 8, 34])
print(b)


# In[5]:


c = np.array([(2, 7, 3), (65, 2, 89)]) # Matriz de 6 elementos.
print(c)


# In[6]:


d = np.array([(2, 7, 3), (9, 2, 6), (5, 3, 2)])
print(d)


# In[7]:


e = np.zeros((4, 3)) # Cria uma matriz de zeros com dimensão de 4 linhas e 3 colunas
print(e)


# In[8]:


f = np.ones((4, 3)) # .array, .zeros e .ones são funções do pacote numpy.
print(f)


# In[9]:


g = np.eye(5) # Cria matriz com 5 linhas e 5 colunas com números 1 na diagonal.
print(g)


# In[10]:


d.max() # Informa o maior número da matriz b criada.


# In[11]:


d.min()


# In[12]:


d.sum() # Soma todos os elementos da matriz.


# In[13]:


d.mean() # Média dos elementos da matriz.


# In[14]:


d.std() # Calcula o desvio padrão.


# In[2]:


# COMO ABRIR ARQUIVOS USANDO O PANDAS

import pandas as pd

dados = pd.read_excel('C:/Users/jeann/Documents/Python/planilha.xlsx')


# In[3]:


dados.head() # Apresenta apenas as primeiras 5 linhas.


# In[4]:


dados.head(7) # Apresenta as 7 linhas.


# In[10]:


# Acessar conjunto de dados do Kagle

### Conjunto de dados: 120 years of Olympic history: athletes and results

dados2 = pd.read_csv('athlete_events.csv') # Abre arquivo em formato csv


# In[11]:


dados2.head(10)


# In[14]:


# INTRODUÇÃO AO PANDAS

import pandas as pd

alunos = {'Nome': ['Ricardo', 'Pedro', 'Angela', 'Cassia'],
         'Nota': [4, 7, 5.5, 9],
         'Aprovado': ['Não', 'Sim', 'Não', 'Sim']}


# In[15]:


### Transformar o dicionário em um data frame

dataframe = pd.DataFrame(alunos)


# In[16]:


print(dataframe)


# In[17]:


objeto1 = pd.Series([1, 6, 9, 10, 5])
print(objeto1)


# In[19]:


import numpy as np
array = np.array([1, 6, 9, 10, 5])
print(array)


# In[21]:


array2 = np.array([(1, 6, 9, 10, 5), (7, 4, 9, 20, 8)])
print(array2)


# In[22]:


# Transformar array em um vetor

objeto2 = pd.Series(array) # Muitos dos comandos usados em dataframes também servem para séries.
print(objeto2)


# In[6]:


# COMANDOS ÚTEIS DO PANDAS

import pandas as pd

alunos = {'Nome': ['Ricardo', 'Pedro', 'Angela', 'Cassia'],
         'Nota': [4, 7, 5.5, 9],
         'Aprovado': ['Não', 'Sim', 'Não', 'Sim']}

dataframe = pd.DataFrame(alunos)

print(dataframe)


# In[7]:


dataframe.head()


# In[25]:


dataframe.shape # Mostra quantas linhas e colunas o data frame apresenta.


# In[26]:


dataframe.describe() # Descreve a média, valores máximos e mínimos, desvio padrão e percentis.


# In[8]:


# FILTRANDO LINHAS E COLUNAS NO PANDAS

import pandas as pd

alunos = {'Nome': ['Ricardo', 'Pedro', 'Angela', 'Cassia'],
         'Nota': [4, 7, 5.5, 9],
         'Aprovado': ['Não', 'Sim', 'Não', 'Sim']}

dataframe = pd.DataFrame(alunos)

print(dataframe)


# In[9]:


dataframe['Aprovado'] # Filtra a coluna 'Aprovado'.


# In[10]:


dataframe.loc[[0]] # Filtra a linha 0 da tabela.


# In[11]:


dataframe.loc[[1]]


# In[12]:


dataframe.loc[[3]]


# In[13]:


dataframe.loc[[0, 3]]


# In[14]:


dataframe.loc[[0, 2, 3]]


# In[16]:


dataframe.loc[0:3] # Filtra as linhas do índice 0 ao 3; usa apenas um colchete.


# In[17]:


dataframe.loc[dataframe['Nome'] == 'Pedro'] # Filtra a coluna Nome e a linha onde tem Pedro.


# In[18]:


dataframe.loc[dataframe['Nota'] >= 7] # Para filtrar notas iguais e acima de 7.


# In[19]:


dataframe.loc[dataframe['Aprovado'] == 'Não'] # Para filtrar alunas não aprovados.


# In[20]:


# MANIPULANDO LINHAS COM O PANDAS

primeiraslinhas = dataframe.loc[0:2] # Cria novo dataframe com as primeiras linhas de outro DF.
print(primeiraslinhas)


# In[21]:


DF_teste = dataframe # Copia do antigo dataframe.
print(DF_teste)


# In[22]:


aprovados_DF = dataframe[dataframe['Nota'] >= 7] # Cria nova tabela com essas condições.
print(aprovados_DF)


# In[24]:


reprovados_DF = dataframe[dataframe['Nota'] < 7] 
print(reprovados_DF)


# In[25]:


reprovados_DF2 = dataframe[dataframe['Aprovado'] != 'Sim'] # Filtra linhas diferente de Sim.
print(reprovados_DF2)


# In[1]:


# MANIPULANDO COLUNAS COM O PANDAS

import pandas as pd

dados = pd.read_csv('athlete_events.csv')


# In[2]:


dados.head(8)


# In[52]:


dados.rename(columns = {'Name':'Nome', 'Sex':'Sexo', 'Age':'Idade'}) # Renomear colunas.


# In[53]:


dados.rename(columns = {'Name':'Nome', 'Sex':'Sexo', 'Age':'Idade'}, inplace = True) # Inplace para não imprimir a tabela.
# Isso facilita de fazer as modificações.


# In[54]:


altura = dados['Height']


# In[31]:


print(altura)


# In[32]:


type(altura)


# In[55]:


dados['Medal'].value_counts() # Contabiliza quantas vezes aparece cada tipo de medalha.


# In[56]:


dados['City'].value_counts() 


# In[57]:


dados['Sexo'].value_counts() 


# In[58]:


dados.describe()


# In[59]:


dados1 = dados.drop(columns=['ID']) # Exclui a coluna ID


# In[60]:


print(dados1)


# In[61]:


dados1.describe()


# In[62]:


# COMO EXCLUIR COLUNAS DO PANDAS

dados.drop('ID', axis = 1, inplace = True)
dados.drop('Games', axis = 1, inplace = True)
dados.drop('NOC', axis = 1, inplace = True)


# In[64]:


dados.head()


# In[5]:


# COMO CRIAR HISTOGRAMAS

import matplotlib.pyplot as plt # Carregar pacote.

dados.hist(column = 'Age', bins = 10) # O número de bins corresponde ao tamanho das colunas. Por exemplo, se existe uma variação
# dos dados que vai de 10 a 90 anos de idade, então o tamanho das barras seria 8, com 80 (90-10) dividido por 10 bins que dá 8.

plt.show() # Para apresentar o gráfico.

# Interpretação dos bins: # 8 é o tamanho da faixa das barras, por exemplo, entre 10 e 18 anos existe um frequência de 
# 20 mil atletas, entre 18 anos e 26 anos existe um frequência de 160 mil atletas, e assim por diante.


# In[6]:


dados.hist(column = 'Age', bins = 100) # O aumento do bins aumenta a amostragem e permite verificar  detalhes de como os 
# dados estão distribuídos. O tamanho das barras passa a ser menor que 1 unidade.

plt.show()


# In[7]:


dados.hist(column = 'Weight', bins = 100) # Histograma dos pesos dos atletas.

plt.show()


# In[10]:


dados.hist(column = 'Height', bins = 100) # Histograma dos pesos dos atletas.

plt.show()


# In[11]:


# CRIANDO BOXPLOT USANDO PYTHON

dados.boxplot('Age')
plt.show()


# In[12]:


dados.boxplot(column = ['Age', 'Height', 'Weight'])
plt.show()


# In[13]:


import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# In[14]:


plt.scatter(x, y) # Gráfico de dispersão.
plt.show()


# In[16]:


import numpy as np

x1 = np.arange(1, 1000, 1) # Cria um conjunto de dados variando de 1 a 1000 com intervalos de 1 unidade.


# In[17]:


plt.plot(x1, x1**2)
plt.show()


# In[18]:


x2 = np.arange(-1000, 1000, 1)
plt.plot(x2, x2**2)
plt.show()


# In[19]:


plt.plot(x2, -x2**3+4)
plt.show()


# In[31]:


print(dados)


# In[32]:


dados1 = dados.loc[dados['Sex'] == 'M'] # Filtrando os dados de sexo masculino
print(dados1)


# In[35]:


dados1['Sex']


# In[45]:


import plotly.express as px
fig = px.scatter(dados1, x = 'Weight',  y = 'Height') 
fig.show()


# In[46]:


import plotly.express as px
fig1 = px.scatter(dados, x = 'Weight',  y = 'Height') # Para os dados que incluem sexo masculino e feminino.
fig1.show()


# In[47]:


masculino = dados.loc[dados['Sex'] == 'M']
print(masculino)


# In[48]:


altura = masculino['Height']
peso = masculino['Weight']


# In[50]:


plt.scatter(altura, peso) 
plt.show()


# In[1]:


# COMO TRABALHAR COM DADOS FALTANTES (MISSING NaN) EM UM DATASET

import pandas as pd

dados = pd.read_csv('athlete_events.csv')


# In[3]:


dados.head()


# In[4]:


dados2 = dados.dropna() # Exclui todos os dados faltantes.
dados2.head()


# In[5]:


dados.shape # Verifica o tamanho do dataset.


# In[6]:


dados2.shape


# In[7]:


enulo = dados.isnull() # Apresenta os dados faltantes com True ou False.
enulo.head(100) # Apresenta as 100 primeiras linhas.


# In[8]:


faltantes = dados.isnull().sum() # Soma todas as quantidades de valores faltantes.


# In[9]:


print(faltantes) # Apresenta a quantidade de dados faltantes em cada variável do dataset.


# In[10]:


faltantes_percentual = (dados.isnull().sum() / len(dados['ID'])) * 100
print(faltantes_percentual)


# In[15]:


## Para substituir dados faltantes é preciso levar em conta se pode substituir por zero ou fazer a média ou mediana.

dados['Medal'].fillna('Nenhuma', inplace = True) # Utiliza a coluna 'Medal' e preenche os dados faltantes por 'Nenhuma'.
dados['Age'].fillna(dados['Age'].mean(), inplace = True) # Utiliza a coluna 'Age' e faz a média das idades dos atletas 
# para os dados faltantes.
dados['Height'].fillna(dados['Height'].mean(), inplace = True)
dados['Weight'].fillna(dados['Weight'].mean(), inplace = True)


# In[16]:


dados.head(100)


# In[17]:


dados.shape


# In[18]:


faltantes_percentual = (dados.isnull().sum() / len(dados['ID'])) * 100
print(faltantes_percentual)


# In[1]:


# PRIMEIRO CÓDIGO DE MACHINE LEARNING COM PYTHON

## Existem duas fases no processo de machine learning, uma fase de treino em que os algoritmos serão
## aprendidos pela máquina e a fase de teste, em que será aplicado a técnica de aprendizado para
## fazer as classificações.

import pandas as pd
arquivo = pd.read_csv("wine_dataset.csv")


# In[3]:


arquivo.head()


# In[4]:


arquivo['style'] = arquivo['style'].replace('red', 0) # Alguns dados são necessários estarem em formato numérico.
arquivo['style'] = arquivo['style'].replace('white', 1)


# In[5]:


arquivo.head(40)


# In[6]:


# Separando as variáveis entre preditora e alvo

y = arquivo['style'] # Variável alvo em que se quer decobrir o tipo de vinho (tinto ou branco).
x = arquivo.drop('style', axis = 1) # Exclusão da variável alvo para classificar todas as variáveis preditoras do dataset.
# Axis = 1 indica o eixo onde estão todas as variáveis (primeira linha).


# In[7]:


print(y)


# In[8]:


print(x)


# In[12]:


# Do pacote model_selection presente no sklearn importar o pacote train_test_split.

from sklearn.model_selection import train_test_split

# Criando conjunto de dados de treino e teste.

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.3) # Faz o teste com 30% dos dados.
# O treino é feito com a maior parte dos dados (70%).


# In[13]:


# Importa o pacote de machine learning

from sklearn.ensemble import ExtraTreesClassifier 

# Criação do modelo

modelo = ExtraTreesClassifier() # Cria o modelo de machine learning.
modelo.fit(x_treino, y_treino) # Aplica o modelo de machine learning aos meus dados de treino.

# Imprimindo os resultados

resultado = modelo.score(x_teste, y_teste) # Após ser feito o treino, o modelo indica a acurácia do teste.
print('Acurácia:', resultado)


# In[15]:


# Puxando algumas amostras aleatórias para verificar o modelo

y_teste[400:406]


# In[16]:


x_teste[400:406]


# In[18]:


# Fazendo as previsões do modelo para identificar as classificações 
# Prever a partir das variáveis preditoras o tipo de vinho (tinto (0) ou branco (1))

previsoes = modelo.predict(x_teste[400:406])


# In[19]:


previsoes


# In[1]:


# CLASSES, OBJETOS, MÉTODOS, HERANÇA, CONSTRUTOR

def teste(v, i):
    valor = v
    incremento = i
    resultado = valor + incremento
    return resultado


# In[2]:


a = teste(10, 1)


# In[3]:


a


# In[4]:


## valor, incremento e resultado são variáveis locais, ou seja, estão dentro da função e por isso
## não retornam um valor e sim um erro. Ao oantrário, a variável 'a' retorna o resultado da função.


# In[6]:


# Classes e métodos

class DidaticaTech: # Descrição da classe que envolve o método.
    def incrementa(self, v, i): # Descrição do método (função). self, v e i são atributos da classe.
        valor = v
        incremento = i
        resultado = valor + incremento
        return resultado


# In[34]:


a = DidaticaTech()


# In[35]:


b = a.incrementa(10, 1)


# In[36]:


b


# In[37]:


a = DidaticaTech().incrementa(10, 1)


# In[38]:


a


# In[39]:


class DidaticaTech: # Descrição da classe que envolve o método.
    def incrementa(self, v, i): # Descrição do método (função). self, v e i são atributos da classe.
        self.valor = v
        self.incremento = i
        self.resultado = self.valor + self.incremento
        return self.resultado
    
# O uso do self permite que as variáveis dentro da função retorne um resultado quando chamadas.


# In[40]:


a = DidaticaTech()


# In[41]:


b = a.incrementa(10, 1)


# In[42]:


b


# In[43]:


a.valor # O uso do self permite que a variável dentro da função seja chamada.


# In[44]:


a.incremento


# In[45]:


class DidaticaTech: 
    def __init__(self, v: int, i: int): # O init permite adicionar uma função adjacente.
        self.valor = v
        self.incremento = i
    def incrementa(self):
        self.valor = self.valor + self.incremento # A função incrementa permitirá a retroalimentação do valor
# que será incrementado a cada chamada da da variável.


# In[46]:


a = DidaticaTech(10, 1) # Define os valores para a função


# In[50]:


a


# In[51]:


a.incrementa()


# In[55]:


a.valor


# In[56]:


a.incrementa() # A função incrementa() sempre deve ser chamada antes.


# In[58]:


a.valor # Então será incrementado mais 1 no valor.


# In[59]:


class DidaticaTech: 
    def __init__(self, v = 10, i = 1): # Os valores podem ser definidos antes.
        self.valor = v
        self.incremento = i
    def incrementa(self):
        self.valor = self.valor + self.incremento # A função incrementa permitirá a retroalimentação do valor
# que será incrementado a cada chamada da da variável.


# In[62]:


a = DidaticaTech() # Como os valores já foram definidos, eles não precisam ser acrescentados nessa função.


# In[63]:


a.incrementa()


# In[64]:


a.valor


# In[65]:


a.incrementa()


# In[66]:


a.valor


# In[67]:


b = DidaticaTech()


# In[68]:


b.incrementa()


# In[69]:


b.valor


# In[70]:


b.incrementa()


# In[71]:


b.valor


# In[72]:


## As variáveis a e b são consideradas objetos.


# In[74]:


class DidaticaTech: 
    def __init__(self, v = 10, i = 1): # Os valores podem ser definidos antes.
        self.valor = v
        self.incremento = i
    def incrementa(self):
        self.valor = self.valor + self.incremento
    def verifica(self):
        if self.valor > 12:
            print("Ultrapassou 12")
        else:
            print("Não ultrapassou 12")
    def exponencial(self, e):
        self.valor_exponencial = self.valor**e
    def incrementa_quadrado(self):
        self.incrementa() # Reultiliza a função incrementa().
        self.exponencial(2) # Reultiliza a função exponencial().


# In[75]:


a = DidaticaTech()


# In[76]:


a.incrementa() # Primeiro chama a função.


# In[77]:


a.valor


# In[78]:


a.incrementa()


# In[79]:


a.valor


# In[80]:


a.incrementa()


# In[81]:


a.valor


# In[82]:


a.verifica()


# In[84]:


a.exponencial(4) # Nesse caso requer o valor e.


# In[85]:


a.valor_exponencial


# In[86]:


a.incrementa_quadrado()


# In[88]:


a.valor


# In[89]:


a.valor_exponencial # Utiliza o valor de 15 e eleva ao quadrado (2).


# In[90]:


# Herança

class Calculos(DidaticaTech): # A função anterior é herdada com uso do 'class'.
    pass


# In[91]:


c = Calculos()


# In[93]:


c.incrementa()


# In[94]:


c.valor


# In[101]:


class Calculos(DidaticaTech):
    def decrementa(self):
        self.valor = self.valor - self.incremento


# In[102]:


c = Calculos()


# In[107]:


c.decrementa()


# In[108]:


c.valor


# In[109]:


c.decrementa()


# In[110]:


c.valor


# In[116]:


# Método construtor init

class Calculos(DidaticaTech):
    def __init__(self, d = 5): # O método construtor com init substitui o self.valor da primeira função, apesar 
        ## ser herdado em Class.
        self.divisor = d
    def decrementa(self):
        self.valor = self.valor - self.incremento
    def divide(self):
        self.valor = self.valor/self.divisor


# In[117]:


c = Calculos()


# In[118]:


c.incrementa()


# In[119]:


class Calculos(DidaticaTech):
    def __init__(self, d = 5):
        super().__init__(v = 10, i = 1) # Chama o método construtor da primeira função DidaticaTech, 
    ## assim não é sobreposto.
        self.divisor = d
    def decrementa(self):
        self.valor = self.valor - self.incremento
    def divide(self):
        self.valor = self.valor/self.divisor


# In[120]:


c = Calculos()


# In[121]:


c.incrementa()


# In[122]:


c.valor


# In[123]:


c.decrementa()


# In[124]:


c.valor


# In[125]:


c.divide()


# In[126]:


c.valor


# In[1]:


# PREVEDO DADOS DIARIAMENTE COM MACHINE LEARNING

import pandas as pd
df = pd.read_csv("GSPC.csv")
df.head()


# In[2]:


df = df.drop('Date', axis = 1) # Retira a coluna de data.


# In[3]:


df[-2::] # Apresenta as duas últimas linhas.


# In[6]:


# A linha 17216 indica o que aconteceu no dia anterior, e a linha 17217 é o que acontece no próximo dia.
# Assim, retiramos a última linha para treinar o modelo e assim poder fazer previsões do dia seguinte.

amanha = df[-1::] # Renomear e guardar a última célula como 'amanhã'.
amanha


# In[7]:


base = df.drop(df[-1::].index, axis = 0) # Nomear a nova base de dados sem a última linha.
base.tail()


# In[10]:


base['target'] = base['Close'][1:len(base)].reset_index(drop = True) # Cria uma nova variável chamada 'target' (alvo) e
# Copia os valores de fechamento da coluna 'close' que são os valores finais/fechados sem variação. Os valores são
# copiados a partir da linha 1 para que possa perver o último valor de fechamento, o qual será substituído por NaN ou
# sendo considerado como o amanhã que será previsto.
# O reset.index serve para não modificar os valores das outras linhas de dados. 
base.tail()


# In[11]:


prev = base[-1::].drop('target', axis = 1) # Renomeia a última linha como 'prev'.
prev


# In[12]:


treino = base.drop(base[-1::].index, axis = 0) # Retira a última linha, que será a previsão do amanhã.
treino.tail() # Agora os valores de 'target' se apresentam como o valor da linha posterior da coluna 'Close',
# até chegar na última linha 17215.


# In[13]:


treino.loc[treino['target'] > treino['Close'], 'target'] = 1 # Substitui os valores de target por 1 caso sejam
# maiores que os valores da coluna 'Close'.
treino.tail()


# In[15]:


treino.loc[treino['target'] != 1, 'target'] = 0 # Substitui os valores diferentes de 1 por 0.
treino.tail() # Agora os valores da variável 'target' são 0 ou 1.


# In[17]:


y = treino['target']
x = treino.drop('target', axis = 1)

from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.3)

from sklearn.ensemble import ExtraTreesClassifier
modelo = ExtraTreesClassifier()
modelo.fit(x_treino, y_treino)

resultado = modelo.score(x_teste, y_teste)
print("Acurácia:", resultado) # Se a acurácia for muito baixa indica que o modelo pode ser inadequado
# para a análise ou os dados são muito imprevisíveis.


# In[18]:


prev # Última linha que foi salva


# In[19]:


modelo.predict(prev) # Linha aplicada ao modelo de previsão.


# In[20]:


prev['target'] = modelo.predict(prev) # Adicionar a coluna 'target' com a previsão do modelo.
prev


# In[21]:


amanha # Agora podemos prever os dados com a linha de 'amanha'.


# In[27]:


base.tail()


# In[28]:


base = base.append(amanha, sort = True) # Adiciona a linha 'amanha' na base com a previsão dos dados.
base.tail()


# In[ ]:


# Colocar o modelo em produção para ser usado com frequência


