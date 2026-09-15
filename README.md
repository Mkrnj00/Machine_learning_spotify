# Inteligencia Musical: Predicción de Popularidad de Canciones en Spotify

**Asignatura:** MLY1101 – Machine Learning **Evaluación:** Evaluación Parcial N°1 – Presentación y defensa técnica del proyecto **Caso:** C – Inteligencia musical y predicción de popularidad de canciones (Spotify Tracks) **Integrantes del grupo:** \[Matías Cornejo, José Filidey, Gerald Santana\] **Fecha:** \[14/09/2026\]

## 1. Descripción del problema de negocio

La plataforma de Spotify necesita anticipar qué canciones tienen mayor probabilidad de ser populares para optimizar decisiones de inversión en marketing y promoción de artistas. Por lo que requieren un análisis exploratorio de los datos, evaluación de sesgos y una serie de conclusiones preliminares para empezar a tomar decisiones sobre su catálogo.

### 1.1 Planteamiento del problema

¿Es posible predecir el nivel de popularidad de una canción a partir de sus características de audio (danceability, energy, tempo, etc.) y metadatos (género, duración, explicit)?

Esta pregunta resume principalmente el objetivo general de los requerimientos del cliente. Puesto que a un nivel general predecir qué canciones priorizar o en que artistas invertir reduce el riesgo de pérdidas y aumenta la probabilidad de ganancias futuras.

## 2. Objetivos del proyecto

### 2.1 Objetivo general

Desarrollar un modelo de Machine Learning capaz de predecir la popularidad de una canción en Spotify a partir de sus atributos de audio y metadatos.

### 2.2 Objetivos específicos

Realizar un análisis exploratorio de datos (EDA) para identificar patrones y relaciones entre las variables de audio y la popularidad.

Evaluar la calidad de los datos e implementar un proceso de limpieza y preparación adecuado.

Identificar las variables más relevantes para la predicción de popularidad.

Evaluar posibles sesgos y consideraciones éticas asociadas al uso de estos datos.

## 3. Estadísticas descriptivas del dataset 

| Artistas únicos | Albumes únicos | Canciones únicas | Popularidad Promedio |
|-----------------|----------------|------------------|----------------------|
| 31 437          | 46 589         | 73 608           | 33.24                |

Para conseguir estos datos se utilizó la variable .unique() y .mean()

![](images/image10.png)

## 4. Descripción de las fuentes de datos

### 4.1 Origen del dataset

- **Nombre:** Spotify Tracks Dataset

- **Fuente:** AVA Duoc

- **Herramientas utilizadas:** \[Google Colab, Google Drive\]

### 4.2 Estructura del dataset

- **N° de registros:** 114.000 filas

- **N° de columnas:** 20 (+ índice)

- **Géneros incluidos:** 114 géneros musicales

### 4.3 Diccionario de variables

| Variable         | Tipo        | Descripción                                    |
|------------------|-------------|------------------------------------------------|
| track_id         | string      | Identificador único de la canción en Spotify   |
| artists          | string      | Artista(s) que interpretan la canción          |
| album_name       | string      | Nombre del álbum                               |
| track_name       | string      | Nombre de la canción                           |
| popularity       | int (0-100) | Puntaje de popularidad asignado por Spotify    |
| duration_ms      | int         | Duración de la canción en milisegundos         |
| explicit         | bool        | Indica si la canción tiene contenido explícito |
| danceability     | float (0-1) | Qué tan bailable es la canción                 |
| energy           | float (0-1) | Intensidad y actividad percibida               |
| key              | int         | Tonalidad musical                              |
| loudness         | float (dB)  | Volumen general en decibeles                   |
| mode             | int (0/1)   | Modalidad (menor/mayor)                        |
| speechiness      | float (0-1) | Presencia de palabras habladas                 |
| acousticness     | float (0-1) | Confianza de que la pista sea acústica         |
| instrumentalness | float (0-1) | Predicción de ausencia de voz                  |
| liveness         | float (0-1) | Presencia de audiencia en vivo                 |
| valence          | float (0-1) | Positividad transmitida por la pista           |
| tempo            | float (BPM) | Tempo estimado                                 |
| time_signature   | int         | Compás estimado                                |
| track_genre      | string      | Género musical asignado                        |

## 5. Preparación y análisis exploratorio de los datos (EDA)

### 5.1 Calidad de los datos

Al momento del inicio del EDA nos percatamos que la cantidad de columnas era correcta y llegamos a la conclusión de que no sobra ninguna, por lo tanto no se modificaron.

![](images/image4.png)

Posteriormente se realizó un búsqueda de valores nulos en el DF, mediante varias funciones se ubicaron 3 datos nulos en una sola fila. Primero se usó la función .info y luego la función .isnull para ubicar de manera más precisa los valores nulos.

Se descubrió que los 3 valores nulos estaban en una misma fila, al ser pocos valores nulos y en una sola fila se decidió borrar la fila ya que no afecta al DF.

Antes de borrar Después de borrar

![](images/image3.png) ![](images/image7.png)

Luego de esto, se realizó la eliminación de la fila que contenía datos nulos y la eliminación de la columna duration_ms , ya que el mapa de correlación mostró una correlación de 1 con duration_min.

![](images/image2.png)

### 5.2 Análisis exploratorio

**EDA:**

Teniendo en cuenta el problema de la empresa, la cual requiere la predicción de la popularidad de las canciones se tomaron en cuenta las variables de Popularidad, los géneros, los artistas y la duración de las canciones.

De esta manera se reunió la cantidad de información suficiente para que la empresa pueda entender cómo se comporta la popularidad para el futuro.

Primero se tomó en cuenta la distribución de la popularidad de las canciones por duración de las mismas. De esta manera se puede visualizar cómo se comporta la popularidad de las canciones según su duración y en donde se concentra la mayoría del catálogo de la empresa.

**Gráfico para observar la distribución de la popularidad de las canciones por el tiempo de duración, para observar outliers.**

![](images/image9.png)

**Top 5 Artistas Mas Populares**

![](images/image8.png)

**Los 5 Géneros Más Populares**

![](images/image11.png)

**Porcentaje de Canciones Explícitas vs No Explícitas (Popularidad > 60)**

![](images/image1.png)

### 5.3 Variables menos relevantes para el modelo

![](images/image6.png)

El análisis de correlación lineal muestra que las variables numéricas , es decir los atributos de audio, por sí solos no explican la popularidad de una canción.

### 5.4 Variables más relevantes para el modelo

![](images/image5.png)

Se agruparon las canciones por género y se calculó la popularidad promedio de cada uno, para identificar los 10 géneros más populares y los 10 menos populares. El boxplot muestra no solo el promedio, sino la dispersión completa de cada género, permitiendo ver si la popularidad es consistente dentro del género o varía mucho entre sus canciones.

## 6. Evaluación de sesgos, ética y privacidad de los datos

### 6.1 Posibles sesgos

Sesgo temporal: Canciones más recientes pueden tener ventaja en el score de popularidad.

Sesgo de sobrerrepresentación: El df cuenta con 1000 canciones por cada género. Al ser una distribución artificial no refleja el consumo real de spotify.

### 6.2 Consideraciones éticas

Riesgo de retroalimentación algorítmica: Predecir y promover solo lo “popular” puede reducir la diversidad musical.

Falta de equidad con los artistas: Spotify cuenta tanto con artistas respaldados por enormes sellos discográficos , como con artistas independientes. Por lo que si predecimos la popularidad sin ese contexto no hay forma de que un artista independiente pueda competir con los números de un artista ya consolidado.

Usar la variable “Explicit” como predictor : Considerar esa variable en decidir qué contenido promocionar podría incentivar contenido explícito solo para ganar visibilidad, sin considerar el impacto en audiencias sensibles.

## 7. Conclusiones preliminares

El análisis muestra que la popularidad depende más del género musical y factores externos que de las características de audio.

**Hallazgo principal**

El género con mayor popularidad promedio (\`pop-film\`, 59.3) supera por más de 57 puntos al más bajo (\`iranian\`, 2.2). Géneros como \`pop\`, \`jazz\` y \`classical\` muestran cajas muy amplias, indicando alta variabilidad interna. Este dato es muy importante, porque no da a entender que el género por sí solo no garantiza popularidad, pero sigue siendo la variable considerando tanto categóricas como numéricas la que mejor separa canciones populares de no populares.
