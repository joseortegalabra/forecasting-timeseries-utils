# Forecasting-Models
 Repo with differents models for forecasting. En cada carpeta se encuentran códigos de forecasting usando diferentes modelos o con diferentes objetivos. Todos pensandos en modelos globales, es decir, un modelo para predecir múltiples series de tiempo


## Carpetas / temas
Cada carpeta contiene códigos para forecasting usando diferentes modelos o variando su enfoque

- **hierarchical:**  modelos para forecasting jerárquico, hacer consolidación de forecasts. Multiples series de tiempo y sus agregaciones (ej series a nivel ciudad y su agregación a nivel ciudad y a nivel país), hacer forecast de todas las series y consolidar los volúmenes forecasteados

- **nixtla-mlforecast:** usar modelos de machine learning para predecir múltiples series de tiempo. Usando framework/librería nixtla para poder hacer forecast por recursividad

- **ml-prophet**: forecast intermedio de series univariadas con prophet (u otro modelo estadístico) y luego realizar stacking de modelos de machine learning con nixtla

- **stacking-specialized:** modelo para forecast global de series donde el modelo se compone de varias modelos donde cada uno aprende de un subconjunto de series (se entrena con un subconjunto de series pero predicen para todas las series, existiendo un capa final en el stacking para ponderar las predicciones). Similar a una lógica de bagging pero sampleando series de tiempo en lugar de observaciones independientes

