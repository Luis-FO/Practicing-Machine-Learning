import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model


dataset = keras.datasets.fashion_mnist
((imagens_treino, identificacoes_treino),(imagens_teste, identificacoes_teste)) = dataset.load_data()

total_de_classificacoes = 10
nomes_de_classificacoes = ['Camiseta', 'Calça', 'Pullover',
                           'Vestido', 'Casaco', 'Sandalia', 'Camisa',
                           'Tenis', 'Bolsa', 'Bota']

# Normalização
imagens_treino = imagens_treino/float(255)
imagens_treino.max()

modelo = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(256, activation = tensorflow.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation = tensorflow.nn.softmax)
])

modelo.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
               metrics = ['accuracy'])

hist = modelo.fit(imagens_treino, identificacoes_treino, epochs = 32, batch_size=32, validation_split=0.2)

#usar nomes mais descritivos como modelo_epochs5_nos3.h5
modelo.save("modelo_epochs5_nos3.h5")
modelo_salvo = load_model("modelo_epochs5_nos3.h5")

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Acurácia por épocas')
plt.xlabel('epocas')
plt.ylabel('acuracia')
plt.legend(['treino', 'validação'])

plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Perda por épocas')
plt.xlabel('epocas')
plt.ylabel('Perda')
plt.legend(['treino', 'validação'])

plt.show()

imagens_teste = imagens_teste/float(255)

modelo.evaluate(imagens_teste, identificacoes_teste, batch_size=1)

testes = modelo.predict(imagens_teste, batch_size=1)
print("resultado teste", np.argmax(testes[1]))
print('número da imagem de teste:', identificacoes_teste[1])