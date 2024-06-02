# Modelos 1 proyecto alternativo

# Miembros del grupo
Jorge Andrés Cardeño Devia, CC 1152220936, Ingeniería de Sistemas


# Acerca de los datos
El conjunto de datos ofrece información completa sobre los factores de salud que influyen en el desarrollo de la osteoporosis, incluyendo detalles demográficos, elecciones de estilo de vida, historial médico e indicadores de salud ósea. Su objetivo es facilitar la investigación en la predicción de la osteoporosis, permitiendo que los modelos de aprendizaje automático identifiquen a las personas en riesgo. Analizar factores como la edad, el género, los cambios hormonales y los hábitos de vida puede ayudar a mejorar la gestión y las estrategias de prevención de la osteoporosis.

# Para descargar los datos manualmente
https://www.kaggle.com/datasets/amitvkulkarni/lifestyle-factors-influencing-osteoporosis

# Fase 2
Para ejecutar el contenedor con éxito:

Imagen del contenedor:
docker build -t models_scripts

Para correr el contenedor:
docker run -it --rm models_scripts

# Fase 3
Para ejecutar los endpoints con éxito:

Descargar las dependencias (Crear ambiente virtual):
python3 -m venv venv
pip install -r requirements.txt

Para correr la aplicacion:
fastapi dev apirest.py
