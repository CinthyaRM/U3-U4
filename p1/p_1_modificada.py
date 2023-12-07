# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Conversión de gigabytes a megabytes
gigabytes = np.array([1, 2, 5, 10, 20], dtype=float)
megabytes = np.array([1024, 2048, 5120, 10240, 20480], dtype=float)

oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
oculta2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1, oculta2, salida])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Comenzando entrenamiento...")
historial = modelo.fit(gigabytes, megabytes, epochs=1000, verbose=False)
print("Modelo entrenado!")

plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
plt.show()

print("Realizar una predicción")
resultado = modelo.predict([30])  # Predicción para 30 gigabytes
print("El resultado es " + str(resultado) + " megabytes")

# Guardar el modelo
modelo.save('gigabytes_a_megabytes.h5')


!ls

!pip install tensorflowjs

!mkdir RiosMartinez

!tensorflowjs_converter --input_format keras gigabytes_a_megabytes.h5 RiosMartinez

!ls RiosMartinez