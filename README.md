# Clasificación de Texturas de Llantas (Tire Texture Image Recognition)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

Este repositorio contiene la implementación y evaluación de modelos de Deep Learning para la detección de defectos (grietas) en texturas de llantas. El proyecto explora y compara el rendimiento de una Red Neuronal Convolucional (CNN) construida desde cero frente a la aplicación de Transfer Learning utilizando la arquitectura **Xception**.

## 📁 Contenido del Repositorio

* `CNN_Tires.ipynb`: Notebook que implementa una CNN personalizada en PyTorch. Incluye un módulo de preprocesamiento avanzado (filtros CLAHE y BlackHat), aumento de datos y un bucle de entrenamiento orientado a optimizar el F1-Score dinámico.
* `Xception_Tires.ipynb`: Notebook que adapta y afina el modelo preentrenado `xception71.tf_in1k` (mediante `timm`). Emplea un entrenamiento estratégico en dos fases (entrenamiento del cabezal y fine-tuning parcial de bloques) logrando un rendimiento superior.

## 📊 Conjunto de Datos
Los modelos fueron entrenados utilizando el dataset público [Tire Texture Image Recognition](https://www.kaggle.com/datasets/jehanbhathena/tire-texture-image-recognition) proveniente de Kaggle. El conjunto de datos clasifica recortes de imágenes de llantas en dos categorías:
* `cracked` (con grietas o defectos estructurales)
* `normal` (en buen estado)

## 🛠️ Tecnologías y Requisitos

Para ejecutar los notebooks se requieren las siguientes herramientas y librerías de Python:
* `torch` y `torchvision` (PyTorch)
* `timm` (PyTorch Image Models para cargar la arquitectura Xception)
* `opencv-python` (Procesamiento de imágenes)
* `scikit-learn` (Evaluación de métricas y división estratificada de datos)
* `grad-cam` (Visualización de mapas de activación y explicabilidad del modelo)
* `matplotlib` y `seaborn` (Visualización de curvas ROC y matrices de confusión)

## 📈 Resultados Principales

Dado que el objetivo crítico del sistema es detectar llantas defectuosas, ambos modelos fueron evaluados maximizando el **F1-Score de la clase 'cracked'** mediante la calibración del umbral de decisión en el conjunto de validación.

Al evaluar en el conjunto de prueba (Test Set), se obtuvieron los siguientes resultados:

| Modelo | Exactitud (Accuracy) | F1-Score (Cracked) | Umbral Óptimo |
| :--- | :---: | :---: | :---: |
| **Xception (Transfer Learning)** | 76% | **0.78** | 0.50 |
| **CNN Personalizada** | 67% | 0.68 | 0.60 |

## 👨‍💻 Autor
* **Carlos Enrique Villanueva Portal** - *Universidad Nacional de Ingeniería (UNI)*
