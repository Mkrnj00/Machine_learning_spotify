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

### 2.3 KPIs del proyecto

Definimos los siguientes indicadores y metas para la futura evaluación del modelo. Como la variable popularity tiene valores entre 0 y 100, se propone predecir primero ese puntaje y evaluar los resultados con métricas de regresion. Para los indicadores que necesitan identificar canciones populares, se usara el criterio operativo popularidad real o predicha ≥ 60, segun corresponda. Esto es una definicion de trabajo, no un resultado del modelo.

| KPI | Definición | Meta propuesta |
|---|---|---|
| Precision entre canciones predichas populares | De las canciones con popularidad predicha ≥ 60, porcentaje que realmente tiene popularidad ≥ 60. | ≥ 40% |
| Equidad entre generos | Diferencia entre el MAE del genero con mayor error y el de menor error, medida en puntos de popularidad. | ≤ 10 puntos |
| Tasa de falsos positivos | De las canciones con popularidad real < 60, porcentaje que el modelo predice con popularidad ≥ 60. | ≤ 15% |
| Error de predicción (MAE) | Diferencia absoluta promedio entre popularidad real y predicha. | Reducir al menos 20% el MAE respecto de predecir el promedio del genero |

### 2.4 Metodologia utilizada (CRISP-DM)

Para desarrollar este proyecto utilizamos la metodología CRISP-DM, que permite organizar las distintas etapas de un proyecto de Machine Learning, desde comprender el problema hasta evaluar los resultados.

| Etapa | Descripción | Estado |
|---|---|---|
| 1. Comprension del negocio | Se identifico la necesidad de analizar y predecir la popularidad de las canciones para apoyar las decisiones de promocion musical. | Realizado |
| 2. Comprension de los datos | Se exploro el dataset de Spotify, revisando sus variables, columnas, valores nulos y relaciones con la popularidad. | Realizado |
| 3. Preparacion de los datos | Se realizo la limpieza de valores nulos, el tratamiento de canciones duplicadas y la revision de valores atipicos para preparar los datos. | Realizado |
| 4. Modelado | Se planifica entrenar y comparar modelos supervisados, además de explorar una técnica no supervisada, en la siguiente etapa. | Pendiente |
| 5. Evaluación | Se evaluarán los modelos utilizando métricas y los KPIs definidos cuando se complete el entrenamiento. | Pendiente |
| 6. Despliegue | En una etapa posterior se podra utilizar el modelo para apoyar decisiones relacionadas con la promocion de canciones. | Pendiente |

En esta primera etapa del proyecto nos enfocamos en comprender el problema, analizar la información y preparar los datos. Esto nos permite contar con una base más confiable para desarrollar posteriormente el modelo predictivo.
 
## 3. Estadísticas descriptivas del dataset

Para conocer mejor la informacion disponible, realizamos un analisis inicial utilizando funciones de Python como `nunique()`, `mean()`, `info()` y `describe()`.

El dataset original contiene 114.000 registros y 21 columnas, incluyendo una columna de índice que no aporta información al analisis.

| Indicador | Resultado |
|---|---|
| Registros originales | 114.000 |
| Artistas únicos | 31.437 |
| Albumes únicos | 46.589 |
| Nombres de canciones unicos | 73.608 |
| Canciones únicas por `track_id` después de eliminar nulos | 89.740 |
| Popularidad promedio del dataset original | 33,24 |
| Generos musicales en el dataset original | 114 |
| Generos musicales después de la limpieza | 113 |

Para contar correctamente las canciones unicas utilizamos `track_id`, ya que diferentes canciones pueden tener el mismo nombre. Ademas, identificamos que algunas canciones aparecen repetidas porque están clasificadas en más de un género musical.

Por lo cual este analisis nos permitio conocer la cantidad de información disponible y detectar registros repetidos. Por eso, fue necesario realizar una limpieza de datos antes de continuar con el análisis de popularidad.,

![](images/image10.png)

## 4. Descripción de las fuentes de datos

### 4.1 Origen del dataset

- **Nombre:** Spotify Tracks Dataset

- **Fuente:** AVA Duoc

- **Justificación de las herramientas utilizadas:**

Para desarrollar el proyecto utilizamos google colab, ya que nos permite trabajar con Python y sus librerias sin necesidad de instalar programas adicionales. Tambien utilizamos github para organizar los archivos, mantener un registro de los cambios y facilitar el trabajo en equipo. Por ultimo, google drive se considero como una alternativa para almacenar y compartir la base de datos.

Estas herramientas nos permiten mantener el proyecto organizado.

  | Herramienta | Uso | Justificacion |
  |--|--|--|
  | Google Colab | Desarrollo del notebook | Herramienta hecha para poder crear notebooks de ciencia de datos con python y mediante el uso de la importacion de librerias, no es necesario descargar nada |
  | GitHub | Versionado y entrega | Para mantener registro del aporte de cada miembro, controlar las versiones del proyecto y poder reproducir el proyecto |

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

Al momento del inicio del EDA, utilizamos Python y la librería Pandas para revisar la informacion. Mediante funciones como info(), describe() e isnull() nos percatamos  que existen valores nulos, registros duplicados y posibles valores atípicos.

**1. Identificacion y eliminacion de valores nulos**

El dataset original contiene 114.000 registros y 21 columnas. Durante la revision encontramos tres valores nulos en las columnas `artists`, `album_name` y `track_name`, correspondientes a una misma fila.

Como se trataba de un solo registro, decidimos eliminarlo, ya que su eliminacion no afecta al conjunto de datos.

Despues de este procedimiento quedaron 113.999 registros.

![](images/image3.png)

![](images/image7.png)

**2. Identificación y tratamiento de duplicados**

Posteriormente revisamos los registros duplicados utilizando la columna `track_id`, que permite identificar cada canción.

Encontramos 24.259 registros con identificadores repetidos. Esto ocurre porque una misma canción puede aparecer clasificada en diferentes géneros musicales.

Para evitar que estas repeticiones afecten los análisis y los resultados de un futuro modelo, eliminamos los duplicados utilizando `drop_duplicates()`, conservando la primera aparición de cada canción.

Después de este proceso quedaron 89.740 canciones únicas.

![](images/image2.png)

Esta decisión también tiene una limitación, ya que algunas canciones pertenecen a más de un género, pero al eliminar los duplicados se conserva solamente el primer género registrado.

**3. Análisis de canciones con popularidad cero**

Durante el analisis identificamos canciones con un valor de popularidad igual a cero.

Despues de la limpieza encontramos 9.447 canciones con popularidad cero (10,53%). Estos valores se concentran de manera diferente segun el genero, por ejemplo en el analisis aparecen proporciones altas en jazz, iranian y romance. Esto no demuestra que sean errores, sino que podrian reflejar caracteristicas de los datos o de como se asigna la popularidad.

Por este motivo decidimos conservar estos registros y tener en cuenta su distribución por género al evaluar los modelos.

**4. Revisión de valores atípicos en la duración**

Tambien analizamos las canciones con duraciones muy cortas o demasiado largas.

Identificamos 790 canciones con duración inferior a un minuto y 550 canciones con duración superior a diez minutos.

Al revisar estos casos, observamos que algunas duraciones pueden ser normales dependiendo del género musical, como canciones infantiles cortas o mezclas de música electrónica más extensas.

Por esta razón decidimos conservar estos registros, ya que eliminarlos podría afectar la representación de ciertos géneros.

**5. Resumen de la preparación de datos**

| Etapa | Registros |
|---|---:|
| Dataset original | 114.000 |
| Después de eliminar valores nulos | 113.999 |
| Después de eliminar duplicados por `track_id` | 89.740 |
| Después de revisar popularidad cero | 89.740 |
| Después de revisar valores atípicos | 89.740 |

La preparacion de datos permitio identificar y corregir problemas que podian afectar el analisis. Finalmente, obtuvimos un conjunto de 89.740 canciones unicas, conservando los valores de popularidad cero y las duraciones atipicas debido a que no encontramos evidencia suficiente para eliminarlos.

### 5.2 Análisis exploratorio de datos (EDA)

En esta etapa analizamos la relación entre la popularidad de las canciones y algunas de sus caracteristicas, como el artista, género musical, contenido explicito y duracion.

El objetivo es identificar patrones que puedan ayudarnos a seleccionar las variables más relevantes para un futuro modelo de Machine Learning.

**1. Relación entre duracion y popularidad**

**Objetivo:** Analizar si la duracion de una cancion tiene relacion con su popularidad e identificar posibles valores atipicos.

![](images/image9.png)

**Hallazgo:** La duracion mediana de las canciones es aproximadamente 3,55 minutos. También encontramos canciones con duraciones extremas, incluyendo algunas que superan los 10 minutos.

**Implicancia:** La duracion por si sola no parece explicar de manera importante la popularidad. Además, decidimos conservar las duraciones atipicas, ya que pueden ser normales en determinados generos musicales.

**2. Top 5 artistas mas populares**

**Objetivo:** Identificar los artistas con mayor popularidad promedio, considerando un minimo de 10 canciones para evitar resultados poco representativos.

![](images/image8.png)

**Hallazgo:** Bad Bunny destaca entre los artistas con mayor popularidad promedio. Para realizar esta comparación se consideraron solamente artistas con al menos 10 canciones.

**Implicancia:** La popularidad puede variar bastante entre artistas. Por eso, es importante considerar que algunos artistas ya tienen una trayectoria y reconocimiento que pueden influir en sus resultados.

**3. Los 5 generos mas populares**

**Objetivo:** Identificar que generos musicales presentan los mayores niveles de popularidad promedio.

![](images/image11.png)

**Hallazgo:** Los generos con mayor popularidad promedio en el analisis actualizado son k-pop, pop-film, metal, chill y latino.

**Implicancia:** El género musical presenta diferencias importantes en los niveles de popularidad, por lo que puede ser una variable relevante para desarrollar un modelo predictivo.

**4. Canciones explicitas y no explicitas**

**Objetivo:** Analizar si existe una relación entre el contenido explicito de una cancion y su popularidad.

![](images/image1.png)

**Hallazgo:** Entre las canciones con popularidad superior a 60, aproximadamente el 14,4% contiene contenido explicito, mientras que en el catalogo analizado este porcentaje es cercano al 8,6%.

**Implicancia:** El contenido explicito podria ser una variable util para el analisis, pero esta relacion no demuestra que una cancion sea mas popular por ser explicita. Tambien debemos considerar los posibles riesgos eticos de utilizar esta caracteristica para recomendar o promocionar musica.

**Conclusion del analisis exploratorio**

Los resultados muestran que existen diferencias importantes de popularidad entre generos y artistas. En cambio, caracteristicas como la duracion no presentan una relacion lineal fuerte con la popularidad.

Estos hallazgos nos permiten identificar variables que podrían ser utiles para el futuro modelo predictivo y comprender mejor cómo se comporta el catalogo musical.

### 5.3 Relacion lineal entre variables numericas y popularidad

Para identificar que caracteristicas tienen relacion con la popularidad, realizamos un analisis de correlacion entre las variables numericas del dataset.

**Objetivo:** Identificar si caracteristicas como la energia, duracion, tempo y otras variables de audio presentan una relacion lineal con la popularidad de las canciones.

![](images/image6.png)

**Hallazgo:** Los resultados muestran que las caracteristicas de audio tienen correlaciones lineales debiles con la popularidad. La variable `instrumentalness` presenta una correlacion aproximada de -0,13, mientras que la duracion tiene una correlacion cercana a -0,02.

Tambien encontramos relaciones entre algunas caracteristicas de audio, como `energy` y `loudness`, con una correlacion aproximada de 0,76.

**Implicancia:** Estos resultados indican que las caracteristicas de audio por si solas no presentan una relacion lineal fuerte con la popularidad. Sin embargo, esto no significa que debamos descartarlas, ya que podrian aportar informacion al combinarlas con otras variables en un modelo de Machine Learning.

### 5.4 Variables mas relevantes para el modelo

Despues de analizar las caracteristicas numericas, realizamos una comparacion de la popularidad segun el genero musical.

**Objetivo:** Identificar si existen diferencias importantes en la popularidad de las canciones dependiendo del genero al que pertenecen.

![](images/image5.png)

**Hallazgo:** El grafico compara los 10 generos con mayor popularidad promedio y los 10 con menor popularidad. Se observan diferencias importantes entre ambos grupos, aunque tambien existe variacion dentro de cada genero.

Esto significa que algunas canciones de generos con baja popularidad promedio pueden alcanzar valores altos, mientras que no todas las canciones de los generos mas populares tienen buenos resultados.

**Implicancia:** El genero musical parece ser una variable relevante para diferenciar los niveles de popularidad de las canciones. Por eso, consideramos importante incluirlo en el futuro modelo predictivo.

**Conclusión:** El analisis exploratorio muestra que el género musical puede influir en las diferencias de popularidad entre canciones. Sin embargo, pertenecer a un genero popular no asegura que una cancion tenga una popularidad alta. Por esta razon, consideramos importante utilizar el género junto con otras características musicales para desarrollar y evaluar los modelos de Machine Learning.

## 6. Evaluacion de sesgos, etica y privacidad de los datos

Durante el desarrollo del proyecto identificamos algunos posibles sesgos y riesgos eticos que pueden afectar los resultados de un futuro modelo de Machine Learning.

Es importante considerar estos aspectos, ya que el objetivo no es solamente predecir la popularidad, sino tambien evitar decisiones que puedan perjudicar a ciertos artistas o generos musicales.

### 6.1 Posibles sesgos

**Sesgo temporal:**

Las canciones mas recientes podrian tener ventajas o desventajas en su nivel de popularidad. Sin embargo, el dataset no contiene fechas de lanzamiento, por lo que no podemos comprobar directamente este sesgo.

Por esta razon, lo consideramos una limitacion del conjunto de datos y no un resultado confirmado.

**Sesgo de representacion de generos:**

El dataset original contiene 1.000 registros por cada uno de los 114 generos musicales. Esta distribucion no necesariamente representa la cantidad real de canciones o reproducciones que tiene cada genero en Spotify.

Ademas, al eliminar las canciones duplicadas por track_id, el numero de generos paso de 114 a 113: el genero songwriter dejo de aparecer. Los generos restantes quedaron con cantidades diferentes, entre 74 y 1.000 canciones por genero.

Esto podria afectar los resultados del futuro modelo, ya que algunos generos podrian tener mayor representacion que otros.

### 6.2 Consideraciones eticas

**Riesgo de retroalimentacion algoritmica:**

Si Spotify utiliza un modelo para promocionar solamente las canciones que tienen mayor popularidad, podria generar un ciclo donde los artistas populares reciben cada vez mas visibilidad, mientras que otros tienen menos oportunidades.

**Desigualdad entre artistas:**

Los artistas reconocidos generalmente cuentan con mas recursos, publicidad y seguidores que los artistas independientes.

Si el modelo utiliza solamente la popularidad historica, podria favorecer a los artistas consolidados y dificultar que nuevos artistas tengan oportunidades de promocion.

**Uso de la variable explicit:**

Durante el analisis encontramos diferencias en la proporcion de canciones explicitas entre el catalogo general y las canciones con mayor popularidad.

Sin embargo, esto no significa que el contenido explicito provoque mayor popularidad. Utilizar esta variable para decidir que canciones promocionar podria generar decisiones poco equilibradas.

### 6.3 Privacidad y proteccion de los datos

El dataset utilizado no contiene informacion personal de los usuarios que escuchan musica, como nombres, correos electronicos o historial individual de reproducciones.

La popularidad corresponde a un indicador general de cada cancion.

Sin embargo, el dataset contiene nombres de artistas, por lo que es importante utilizar esta informacion de manera responsable y solamente para los objetivos academicos del proyecto.

Para considerar la proteccion de los datos, tomamos como referencia:

- **Ley 19.628:** normativa chilena relacionada con la proteccion de datos personales.
- **Ley 21.719:** nueva normativa chilena que modifica el marco de proteccion de datos personales.
- **GDPR:** reglamento europeo que establece principios relacionados con el uso responsable y la proteccion de datos.

En caso de utilizar esta informacion con fines comerciales, tambien seria necesario revisar los terminos y condiciones de uso de los datos de Spotify.

Identificar estos posibles sesgos y riesgos nos permite comprender que los resultados del modelo podrian afectar de manera diferente a los artistas y generos musicales. Por eso, es importante evaluar estos aspectos antes de utilizar las predicciones para tomar decisiones comerciales.

## 7. Conclusiones del proyecto

Durante el desarrollo de este proyecto realizamos un analisis exploratorio del dataset de Spotify para identificar que caracteristicas pueden estar relacionadas con la popularidad de las canciones.

Los resultados nos permitieron comprender mejor el comportamiento de los datos, identificar problemas de calidad y reconocer variables que podrian ser importantes para un futuro modelo de Machine Learning.

### 7.1 Principales hallazgos

**1. Importancia del genero musical**

El analisis mostro que existen diferencias importantes en la popularidad promedio de los generos musicales. Esto nos permite considerar el genero como una variable relevante para el futuro modelo predictivo.

Sin embargo, pertenecer a un genero popular no garantiza que todas sus canciones tengan una popularidad alta.

**2. Caracteristicas de audio**

Las variables de audio, como energia, tempo, duracion y otras caracteristicas, presentaron correlaciones lineales debiles con la popularidad.

Esto indica que ninguna de estas variables, por si sola, permite explicar completamente la popularidad de una cancion.

**3. Calidad y preparacion de los datos**

Durante la preparacion identificamos valores nulos, canciones duplicadas y valores atipicos.

Despues de realizar la limpieza, el dataset quedo con 89.740 canciones unicas, lo que permite trabajar con informacion mas organizada para las siguientes etapas.

**4. Sesgos y aspectos eticos**

Identificamos posibles sesgos relacionados con la representacion de generos, la popularidad de artistas consolidados y la falta de informacion temporal.

Tambien consideramos aspectos de privacidad y los posibles efectos de utilizar las predicciones para decidir que canciones promocionar.

### 7.2 Cumplimiento de los objetivos especificos

| Objetivo especifico | Resultado obtenido | Cumplimiento |
|---|---|---|
| Realizar un analisis exploratorio para identificar patrones | Se identificaron diferencias de popularidad entre generos y relaciones debiles con las variables de audio. | Cumplido |
| Evaluar la calidad y preparar los datos | Se eliminaron valores nulos y duplicados, ademas de revisar los valores atipicos y la popularidad cero. | Cumplido |
| Identificar variables relevantes para la prediccion | El genero musical mostro diferencias importantes de popularidad, mientras que las variables de audio presentaron correlaciones lineales debiles. | Cumplido en etapa exploratoria |
| Evaluar sesgos y consideraciones eticas | Se identificaron posibles sesgos, riesgos eticos y aspectos relacionados con la privacidad. | Cumplido en etapa exploratoria |

### 7.3 Conclusion general

Como conclusion, consideramos que el proyecto nos permitio comprender mejor los factores que pueden estar relacionados con la popularidad de las canciones en Spotify.

Uno de los principales hallazgos fue que el genero musical presenta diferencias importantes en los niveles de popularidad, mientras que las caracteristicas de audio tienen relaciones lineales mas debiles.

Ademas, la limpieza de los datos fue necesaria para evitar que los registros duplicados afectaran los resultados.

Aunque logramos cumplir los objetivos de esta primera etapa, todavia queda pendiente entrenar y evaluar un modelo de Machine Learning que permita comprobar si estas variables realmente ayudan a predecir la popularidad.

Por lo tanto, el siguiente paso sera utilizar el dataset preparado para desarrollar el modelo predictivo y evaluar sus resultados mediante los KPIs definidos anteriormente.
