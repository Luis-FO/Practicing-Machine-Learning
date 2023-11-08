import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


dataset = keras.datasets.fashion_mnist
((imagens_treino, identificacoes_treino),(imagens_teste, identificacoes_teste)) = dataset.load_data()

total_de_classificacoes = 10
nomes_de_classificacoes = ['Camiseta', 'Calça', 'Pullover',
                           'Vestido', 'Casaco', 'Sandalia', 'Camisa',
                           'Tenis', 'Bolsa', 'Bota']

# Normalização
imagens_treino = imagens_treino/float(255)
imagens_treino.max()