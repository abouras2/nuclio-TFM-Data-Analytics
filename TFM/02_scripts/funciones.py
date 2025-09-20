# -*- coding: utf-8 -*-
"""funciones.ipynb

## Importación de librerías
"""

# Librerías estándar
import os
import warnings

# Manipulación de datos
import pandas as pd
import numpy as np

# Configuración de warnings
warnings.filterwarnings('ignore')

# Análisis de nulos
import missingno as msno

# Visualización de datos
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Textos
import unicodedata
#from fuzzywuzzy import process
import re

"""## Funciones

#### exploracion_inicial
"""

def exploracion_inicial(df, nombre=None):
    """
    Realiza una exploración inicial de un DataFrame y muestra información clave.

    Parámetros:
    df (pd.DataFrame): El DataFrame a explorar.
    tipo (str, opcional): El tipo de exploración. 'simple' muestra menos detalles.

    Imprime:
    Información relevante sobre el DataFrame, incluyendo filas, columnas, tipos de datos,
    estadísticas descriptivas, y valores nulos.
    """

    # Información básica sobre el DataFrame
    num_filas, num_columnas = df.shape
    print(f"¿Cuántas filas y columnas hay en el conjunto de datos?")
    print(f"\tHay {num_filas:,} filas y {num_columnas:,} columnas.")
    print('#' * 90)

    # Exploración completa
    print("¿Cuáles son las primeras cinco filas del conjunto de datos?")
    display(df.head())
    print('-' * 100)

    print("¿Cuáles son las últimas cinco filas del conjunto de datos?")
    display(df.tail())
    print('-' * 100)

    print("¿Cómo puedes obtener una muestra aleatoria de filas del conjunto de datos?")
    display(df.sample(n=5))
    print('-' * 100)

    print("¿Cuáles son las columnas del conjunto de datos?")
    print("\n".join(f"\t- {col}" for col in df.columns))
    print('-' * 100)

    print("¿Cuál es el tipo de datos de cada columna?")
    print(df.dtypes)
    print('-' * 100)

    print("¿Cuántas columnas hay de cada tipo de datos?")
    print(df.dtypes.value_counts())
    print('-' * 100)

    print("¿Cómo podríamos obtener información más completa sobre la estructura y el contenido del DataFrame?")
    print(df.info())
    print('-' * 100)

    print("¿Cuántos valores únicos tiene cada columna?")
    print(df.nunique())
    print('-' * 100)

    print("¿Cuáles son los valores únicos de cada columna?")
    df_valores_unicos = pd.DataFrame(df.apply(lambda x: x.unique()))
    display(df_valores_unicos)
    print('-' * 100)

    print("¿Cuáles son las estadísticas descriptivas básicas de todas las columnas?")
    display(df.describe(include='all').fillna(''))
    print('-' * 100)

    print("¿Cuántos valores nulos hay en cada columna del DataFrame?")
    display(df.isnull().sum())
    print('-' * 100)

    print("¿Cuál es el porcentaje de valores nulos por columna, ordenado de mayor a menor?")
    df_nulos = df.isnull().sum().div(len(df)).mul(100).round(2).reset_index().rename(columns = {'index': 'Col', 0: 'pct'})
    df_nulos = df_nulos.sort_values(by = 'pct', ascending=False).reset_index(drop = True)
    display(df_nulos)
    print('-' * 100)

    print("## Valores nulos: Visualización")
    msno.bar(df, figsize = (6, 3), fontsize= 9)
    plt.show()
    print('-' * 100)


    print("## Visualización de patrones en valores nulos")
    msno.matrix(df, figsize = (6, 3), fontsize= 9, sparkline = False)
    plt.show()
    print('-' * 100)

    '''
    msno.heatmap(df, figsize = (6, 3), fontsize= 9)
    plt.show()
    print('-' * 100)
    '''

print('#' * 90)

"""### analizar_columnas"""

def analizar_columnas(df):
    # Crear el DataFrame auxiliar con tipo de datos, número de nulos y porcentaje de nulos
    aux_tipo_nulos = pd.concat([df.dtypes,
                                df.isnull().sum(),
                                df.isnull().sum().div(len(df)).mul(100).round(2)
                               ], axis=1, keys=['Tipo', 'Num_nulos', 'Pct_nulos'])

    # Ordenar el DataFrame auxiliar por porcentaje de nulos de mayor a menor
    aux_tipo_nulos = aux_tipo_nulos.sort_values(by='Pct_nulos', ascending=False)

    # Inicializar las listas
    columnas_numericas = []
    columnas_categoricas = []
    columnas_con_nulos = []
    columnas_valores = []

    # Recorrer el DataFrame para clasificar las columnas
    for columna, info in aux_tipo_nulos.iterrows():
        if info['Tipo'] == 'object':  # Si es tipo categórico
            columnas_categoricas.append(columna)
        elif info['Tipo'] in ['int64', 'float64']:  # Si es numérico
            columnas_numericas.append(columna)

        if info['Num_nulos'] > 0:  # Si hay nulos
            columnas_con_nulos.append(columna)

        valores = np.random.choice(df[columna].unique(), size=min(5, df[columna].nunique()), replace=False).tolist()
        columnas_valores.append(valores)

    aux_tipo_nulos['Valores'] = columnas_valores
    display(aux_tipo_nulos)

    return aux_tipo_nulos#columnas_numericas, columnas_categoricas, columnas_con_nulos

"""### normalizar_texto"""

def normalizar_texto(df, columnas):
    """
    Limpia columnas de texto en un DataFrame:
    - Pasa a minúsculas
    - Elimina tildes/acentos
    - Elimina espacios antes y después

    Parámetros:
    df : pd.DataFrame
        DataFrame de entrada
    columnas : list
        Lista de nombres de columnas a limpiar

    Retorna:
    pd.DataFrame
        DataFrame con las columnas especificadas normalizadas
    """
    for col in columnas:
        df[col] = df[col].str.lower()
        df[col] = df[col].apply(lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8'))
        df[col] = df[col].str.strip()
    return df

#normalizar_texto(df, columnas=['ciudad', 'barrio'])

"""**Definición:**  
La función `exploracion_inicial` realiza una inspección preliminar de un DataFrame, proporcionando información básica y detallada sobre su estructura, tipos de datos y valores nulos. Dependiendo del valor del parámetro `tipo`, se puede elegir entre una exploración completa o una versión resumida.

**Parámetros:**
- `df`: El DataFrame que se desea explorar.
- `tipo`: Un parámetro opcional que determina el nivel de detalle de la exploración. Si `tipo` es `'version_lite'`, la función ejecuta una versión resumida; de lo contrario, muestra un análisis completo.

**Utilidad:**  
La función permite entender rápidamente la estructura y el contenido del conjunto de datos, ayudando a identificar problemas como valores nulos y revisar las primeras y últimas filas, tipos de datos, y estadísticas descriptivas, lo cual es útil para definir los próximos pasos en el análisis.

### deteccion_outliers
"""

def deteccion_outliers (df, variable):
    # Suponiendo que tienes un DataFrame df y quieres analizar la columna 'columna_de_interes'
    columna = df[variable]

    sns.boxplot(
      data=df,
      y=variable,
    )
    plt.show()

    Q1 = columna.quantile(0.25)
    Q3 = columna.quantile(0.75)
    IQR = Q3 - Q1

    print('Valor del segundo cuartil (25%): {:.2f}'.format(Q1))
    print('Valor del tercer cuartil (75%): {:.2f}'.format(Q3))
    print('Valor del rango intercuartil (IQR): {:.2f}'.format(IQR))

    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    print(f"Los valores atípicos se definen como aquellos que caen fuera del siguiente rango:")
    print(f"\t - Límite inferior (considerado extremadamente bajo): {limite_inferior:.2f}")
    print(f"\t - Límite superior (considerado extremadamente alto): {limite_superior:.2f}")


    outilers = list(columna[((columna < limite_inferior) | (columna > limite_superior))].index)
    num_outliers = len(outilers)
    print(f"Hay {num_outliers} outliers en la variable '{variable}'")
    return outilers

"""**Definición:**  
La función `deteccion_outliers` identifica valores atípicos (outliers) en una columna específica de un DataFrame, calculando los límites de normalidad según el rango intercuartílico (IQR) y mostrando aquellos valores que caen fuera de este rango.

**Parámetros:**
- `df`: El DataFrame que contiene los datos a analizar.
- `variable`: La columna en el DataFrame en la que se buscarán los valores atípicos.

**Utilidad:**  
La función ayuda a detectar y visualizar outliers en los datos, lo que permite identificar valores extremos o inusuales que podrían afectar el análisis o el rendimiento de los modelos. Esto es útil para decidir si se deben ajustar, eliminar o estudiar más a fondo estos valores atípicos.

### deteccion_outliers_varias
"""

def deteccion_outliers_varias(df, list_variable):
    """
    Detecta outliers en múltiples variables numéricas de un DataFrame usando el método IQR.
    Para cada variable:
      - Muestra un boxplot
      - Imprime Q1, Q3, IQR y límites
      - Devuelve un diccionario con {nombre_variable: lista_de_indices_outliers}

    Parámetros:
    df (pd.DataFrame): El DataFrame a analizar
    list_variable (list): Lista con nombres de columnas numéricas a analizar

    Retorna:
    dict: Diccionario con {variable: lista_indices_outliers}
    """
    resultados = {}

    for variable in list_variable:
        columna = df[variable]

        # Boxplot
        sns.boxplot(data=df, y=variable)
        plt.show()

        # Cálculo IQR
        Q1 = columna.quantile(0.25)
        Q3 = columna.quantile(0.75)
        IQR = Q3 - Q1

        print(f"Variable: {variable}")
        print('Valor del segundo cuartil (25%): {:.2f}'.format(Q1))
        print('Valor del tercer cuartil (75%): {:.2f}'.format(Q3))
        print('Valor del rango intercuartil (IQR): {:.2f}'.format(IQR))

        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR

        print(f"Los valores atípicos se definen como aquellos que caen fuera del siguiente rango:")
        print(f"\t - Límite inferior: {limite_inferior:.2f}")
        print(f"\t - Límite superior: {limite_superior:.2f}")

        # Detección outliers
        outliers = list(columna[(columna < limite_inferior) | (columna > limite_superior)].index)
        num_outliers = len(outliers)
        print(f"Hay {num_outliers} outliers en la variable '{variable}'")
        print("-" * 80)

        # Guardar en el diccionario con la variable como clave
        resultados[variable] = outliers

    return resultados

"""**Definición:**  
La función `deteccion_outliers_varias` actua igual que deteccion_outliers para una lista de variables en lugar de una sola variable al agregar un loop sobre la lista.
La función `deteccion_outliers_varias` identifica valores atípicos (outliers) en una columna específica de un DataFrame, calculando los límites de normalidad según el rango intercuartílico (IQR) y mostrando aquellos valores que caen fuera de este rango.

**Parámetros:**
- `df`: El DataFrame que contiene los datos a analizar.
- `variable`: La columna en el DataFrame en la que se buscarán los valores atípicos.

**Utilidad:**  
La función ayuda a detectar y visualizar outliers en los datos, lo que permite identificar valores extremos o inusuales que podrían afectar el análisis o el rendimiento de los modelos. Esto es útil para decidir si se deben ajustar, eliminar o estudiar más a fondo estos valores atípicos.

### procesar_fecha
"""

def try_parse_date(date_str):
    try:
        return parser.parse(date_str)
    except (ValueError, TypeError):
        return None

"""**Definición:**  
La función `try_parse_date` intenta convertir una cadena de texto en un objeto de fecha utilizando `dateutil.parser.parse()`. Si la conversión falla debido a un formato inválido o un valor nulo, la función devuelve `None` en lugar de generar un error.  

**Parámetros:**  
- `date_str`: Una cadena de texto que representa una fecha en cualquier formato compatible con `dateutil.parser.parse()`.  

**Utilidad:**  
Esta función es útil cuando se trabaja con datos que contienen fechas en múltiples formatos o cuando existen valores nulos o incorrectos. Evita errores en el procesamiento y permite manejar fechas de manera más robusta en análisis de datos o transformación de información.

### graficar_boxplot_px
"""

def graficar_boxplot_px(df, variable_analisis):
    # Crear el boxplot usando Plotly Express
    fig = px.box(df, y=variable_analisis)

    # Actualizar títulos del gráfico
    fig.update_layout(title=f'Boxplot: {variable_analisis}',
                      yaxis_title='Frecuencia')

    # Actualizar el fondo del gráfico a blanco
    fig.update_layout({
        'plot_bgcolor': 'rgba(255, 255, 255, 1)',
        'xaxis': {'showgrid': True, 'gridcolor': 'lightgrey'},
        'yaxis': {'showgrid': True, 'gridcolor': 'lightgrey'}
    })

    # Mostrar el gráfico
    fig.show()

"""**Definición:**  
La función `graficar_boxplot_px` genera un boxplot utilizando la librería Plotly Express para visualizar la distribución de una variable en un DataFrame, ajustando además el diseño del fondo y los títulos del gráfico.

**Parámetros:**
- `df`: El DataFrame que contiene los datos.
- `variable_analisis`: El nombre de la columna en el DataFrame para la cual se desea generar el boxplot.

**Utilidad:**  
Esta función facilita la creación de gráficos boxplot estéticamente personalizables y claros, ayudando a identificar la dispersión y los posibles valores atípicos en los datos de la variable especificada.

### graficar_boxplot_bivariable_px
"""

def graficar_boxplot_bivariable_px (df, variable_analisis, variable_categorica):
    # Crear el boxplot usando Plotly Express
    fig = px.box(df, x=variable_categorica, y=variable_analisis, color=variable_categorica)

    # Actualizar títulos del gráfico
    fig.update_layout(title=f'Boxplot de {variable_analisis} por {variable_categorica}',
                      xaxis_title=variable_categorica,
                      yaxis_title=variable_analisis)

    # Actualizar el fondo del gráfico a blanco
    fig.update_layout({
        'plot_bgcolor': 'rgba(255, 255, 255, 1)',
        'xaxis': {'showgrid': True, 'gridcolor': 'lightgrey'},
        'yaxis': {'showgrid': True, 'gridcolor': 'lightgrey'}
    })

    # Mostrar el gráfico
    fig.show()

"""**Definición:**  
La función `graficar_boxplot_bivariable_px` genera un boxplot utilizando la librería Plotly Express para visualizar la distribución de una variable en función de una categoría específica, ajustando además el diseño del fondo y los títulos del gráfico.

**Parámetros:**
- `df`: El DataFrame que contiene los datos.
- `variable_analisis`: El nombre de la columna en el DataFrame para la cual se desea generar el boxplot.
- `variable_categorica`: El nombre de la columna categórica que se utilizará en el eje x para segmentar los datos.

**Utilidad:**  
Esta función permite crear gráficos boxplot que muestran la relación entre una variable cuantitativa y una variable categórica, facilitando la identificación de la dispersión, tendencias y posibles valores atípicos en los datos según la categoría especificada.

### graficar_histograma_px
"""

def graficar_histograma_px (df, variable_analisis):
    fig = px.histogram(df, x=variable_analisis, nbins=20,
                       title=f'Distribución de: {variable_analisis}')

    # Calcular media y mediana
    mean_val = df[variable_analisis].mean()
    median_val = df[variable_analisis].median()

    # Añadir línea vertical para la media
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                  annotation_text=f"Media: {mean_val:.2f}", annotation_position="top right")

    # Añadir línea vertical para la mediana
    fig.add_vline(x=median_val, line_dash="dot", line_color="green",
                  annotation_text=f"Mediana: {median_val:.2f}", annotation_position="top left")

    fig.update_layout(xaxis_title=variable_analisis, yaxis_title='Frecuencia')
    fig.show()

"""**Definición:**  
La función `graficar_histograma_px` genera un histograma utilizando la librería Plotly Express para visualizar la distribución de una variable en un DataFrame, incluyendo líneas que representan la media y la mediana.

**Parámetros:**
- `df`: El DataFrame que contiene los datos.
- `variable_analisis`: El nombre de la columna en el DataFrame para la cual se desea generar el histograma.

**Utilidad:**  
Esta función permite crear histogramas que muestran la frecuencia de valores de una variable cuantitativa, facilitando la comprensión de su distribución y la identificación de tendencias centrales mediante la inclusión de líneas para la media y la mediana.

### graficar_histograma_bivariable_px
"""

def graficar_histograma_bivariable_px (df, variable_analisis, variable_categorica=None, bins=20, show_mean_median=True):
    # Crear el histograma con la opción de segmentar por variable categórica
    if variable_categorica:
        fig = px.histogram(df, x=variable_analisis, color=variable_categorica, nbins=bins,
                           title=f'Distribución de {variable_analisis} por {variable_categorica}')
    else:
        fig = px.histogram(df, x=variable_analisis, nbins=bins,
                           title=f'Distribución de: {variable_analisis}')

    # Opcional: Calcular y mostrar líneas de media y mediana
    if show_mean_median:
        mean_val = df[variable_analisis].mean()
        median_val = df[variable_analisis].median()

        # Añadir línea vertical para la media
        fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                      annotation_text=f"Media: {mean_val:.2f}", annotation_position="top right")

        # Añadir línea vertical para la mediana
        fig.add_vline(x=median_val, line_dash="dot", line_color="green",
                      annotation_text=f"Mediana: {median_val:.2f}", annotation_position="top left")

    # Actualizar títulos del gráfico
    fig.update_layout(xaxis_title=variable_analisis, yaxis_title='Frecuencia',
                      plot_bgcolor='rgba(255, 255, 255, 1)',
                      xaxis_showgrid=True, xaxis_gridcolor='lightgrey',
                      yaxis_showgrid=True, yaxis_gridcolor='lightgrey')

    # Mostrar el gráfico
    fig.show()

"""**Definición:**  
La función `graficar_histograma_bivariable_px` genera un histograma utilizando la librería Plotly Express, permitiendo la visualización de la distribución de una variable en función de una variable categórica, con opciones para mostrar líneas de media y mediana.

**Parámetros:**
- `df`: El DataFrame que contiene los datos.
- `variable_analisis`: El nombre de la columna en el DataFrame para la cual se desea generar el histograma.
- `variable_categorica`: (Opcional) El nombre de la columna categórica que se utilizará para segmentar el histograma.
- `bins`: (Opcional) El número de bins (intervalos) en el histograma, con un valor predeterminado de 20.
- `show_mean_median`: (Opcional) Un booleano que indica si se deben mostrar las líneas de media y mediana, con un valor predeterminado de `True`.

**Utilidad:**  
Esta función permite crear histogramas que muestran la distribución de una variable cuantitativa, con la capacidad de segmentar por una variable categórica, facilitando el análisis de datos. También proporciona información adicional sobre la tendencia central a través de las líneas de media y mediana, lo que ayuda a interpretar mejor la distribución de los datos.

### graficar_barras_titulo_px
"""

def graficar_barras_titulo_px(df, variable_analisis, titulo):
    # Contar la frecuencia de la variable de análisis
    volumen = df[variable_analisis].value_counts().reset_index()
    volumen.columns = [variable_analisis, 'Volumen']

    # Calcular el porcentaje
    total = volumen['Volumen'].sum()
    volumen['Porcentaje'] = volumen['Volumen'] / total * 100

    # Formatear el texto: valor absoluto con separador + porcentaje
    volumen['Texto'] = volumen['Volumen'].apply(lambda x: f'{x:,.0f}') + \
                       volumen['Porcentaje'].apply(lambda x: f' ({x:.1f}%)')

    # Crear el gráfico de barras
    fig = px.bar(volumen, x=variable_analisis, y='Volumen', text='Texto')
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(
        title_text=titulo,
        xaxis_title=variable_analisis,
        yaxis_title='Volumen',
        xaxis={'categoryorder': 'total descending'},
        plot_bgcolor='rgba(255, 255, 255, 1)',
        xaxis_showgrid=True,
        yaxis_showgrid=True,
        xaxis_gridcolor='lightgrey',
        yaxis_gridcolor='lightgrey'
    )

    # Forzar eje Y desde 0
    fig.update_yaxes(range=[0, None], tick0=0)

    fig.show()

"""**Definición:**  
La función `graficar_barras_px` genera un gráfico de barras utilizando la librería Plotly Express para visualizar la frecuencia de una variable categórica en un DataFrame.

**Parámetros:**
- `df`: El DataFrame que contiene los datos.
- `variable_analisis`: El nombre de la columna en el DataFrame cuya frecuencia se desea representar en el gráfico de barras.

**Utilidad:**  
Esta función permite crear gráficos de barras que muestran la frecuencia de los valores de una variable categórica, facilitando la comparación visual entre categorías. El gráfico incluye etiquetas de volumen, lo que mejora la interpretación de los datos y ayuda a identificar tendencias en la distribución de la variable analizada.

### graficar_barras_relativo_px
"""

def graficar_barras_relativo_px(df, variable_analisis):
    """
    Definición:
    La función graficar_barras_relativo_px genera un gráfico de barras utilizando la librería Plotly Express para visualizar la frecuencia relativa (porcentaje) de una variable categórica en un DataFrame.

    Parámetros:
    - df: El DataFrame que contiene los datos.
    - variable_analisis: El nombre de la columna en el DataFrame cuya frecuencia relativa se desea representar en el gráfico de barras.

    Utilidad:
    Esta función permite crear gráficos de barras que muestran la distribución porcentual de una variable categórica, facilitando la comparación visual entre categorías. El gráfico incluye etiquetas con los porcentajes, mejorando la interpretación de los datos y ayudando a identificar tendencias en la distribución de la variable analizada.
    """

    # Calcular la frecuencia relativa de la variable de análisis
    volumen = df[variable_analisis].value_counts(normalize=True).reset_index()
    volumen.columns = [variable_analisis, 'Porcentaje']
    volumen['Porcentaje'] *= 100  # Convertir a porcentaje

    # Crear el gráfico de barras
    fig = px.bar(volumen, x=variable_analisis, y='Porcentaje', text=volumen['Porcentaje'].apply(lambda x: f'{x:.2f}%'))
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(title_text=f'Gráfico de barras relativo: {variable_analisis}',
                        xaxis_title=variable_analisis,
                        yaxis_title='Porcentaje',
                        xaxis={'categoryorder': 'total descending'})

    # Actualizar el fondo del gráfico a blanco
    fig.update_layout({
        'plot_bgcolor': 'rgba(255, 255, 255, 1)',
        'xaxis': {'showgrid': True, 'gridcolor': 'lightgrey'},
        'yaxis': {'showgrid': True, 'gridcolor': 'lightgrey'}
    })

    fig.show()

"""### graficar_barras_porcentaje_px"""

def graficar_barras_porcentaje_px(df, variable_analisis):
    # Contar la frecuencia de la variable de análisis
    volumen = df[variable_analisis].value_counts().reset_index()
    volumen.columns = [variable_analisis, 'Volumen']

    # Calcular el total y convertir a porcentaje
    total = volumen['Volumen'].sum()
    volumen['Porcentaje'] = ((volumen['Volumen'] / total) * 100).round(2)

    # Crear el gráfico de barras
    fig = px.bar(volumen, x=variable_analisis, y='Porcentaje', text='Porcentaje')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')  # Mostrar como porcentaje
    fig.update_layout(title_text=f'Gráfico de barras: {variable_analisis}',
                      xaxis_title=variable_analisis,
                      yaxis_title='Porcentaje',
                      xaxis={'categoryorder': 'total descending'})

    # Actualizar el fondo del gráfico a blanco
    fig.update_layout({
        'plot_bgcolor': 'rgba(255, 255, 255, 1)',
        'xaxis': {'showgrid': True, 'gridcolor': 'lightgrey'},
        'yaxis': {'showgrid': True, 'gridcolor': 'lightgrey'}
    })
    display(volumen)
    fig.show()

"""**Definición:**  
La función `graficar_barras_porcentaje_px` genera un gráfico de barras utilizando la librería Plotly Express para visualizar el porcentaje de una variable categórica en un DataFrame.

**Parámetros:**
- `df`: El DataFrame que contiene los datos.
- `variable_analisis`: El nombre de la columna en el DataFrame cuya frecuencia se desea representar en el gráfico de barras.

**Utilidad:**  
Esta función permite crear gráficos de barras que muestran el porcentaje de los valores de una variable categórica, facilitando la comparación visual entre categorías. El gráfico incluye etiquetas de porcentaje, lo que mejora la interpretación de los datos y ayuda a identificar tendencias en la distribución de la variable analizada. Además, proporciona un fondo limpio y una configuración visual que mejora la legibilidad del gráfico.

### graficar_pie_chart
"""

def graficar_pie_chart(df, variable_analisis):
  df_ = df[variable_analisis].value_counts().reset_index()
  df_.columns = [
      variable_analisis,
      'Volumen'
  ]

  fig = px.pie(
      df_,
      names=variable_analisis,
      values='Volumen',
      title=variable_analisis,
      width=800,
      height=500
  )
  fig.show()

"""**Definición:**  
La función `graficar_pie_chart` genera un gráfico de pastel utilizando la librería Plotly Express para visualizar la distribución de frecuencias de una variable categórica en un DataFrame.

**Parámetros:**
- `df`: El DataFrame que contiene los datos.
- `variable_analisis`: El nombre de la columna en el DataFrame cuya distribución se desea representar en el gráfico de pastel.

**Utilidad:**  
Esta función permite crear gráficos de pastel que muestran cómo se distribuyen las frecuencias de los valores de una variable categórica, facilitando la comprensión de la proporción de cada categoría en el conjunto de datos. Es útil para visualizar la composición de un conjunto de datos en un formato claro y conciso.

### graficar_correlacion
"""

def graficar_correlacion(df, variable_x, variable_y):
    # Crear el gráfico de dispersión usando Plotly Express para visualizar la correlación
    fig = px.scatter(df, x=variable_x, y=variable_y,
                     trendline='ols',  # Añade una línea de regresión
                     labels={variable_x: variable_x, variable_y: variable_y},
                     title=f'Correlación entre {variable_x} y {variable_y}')

    # Actualizar el fondo del gráfico a blanco y ajustar la cuadrícula
    fig.update_layout({
        'plot_bgcolor': 'rgba(255, 255, 255, 1)',
        'xaxis': {'showgrid': True, 'gridcolor': 'lightgrey'},
        'yaxis': {'showgrid': True, 'gridcolor': 'lightgrey'}
    })

    # Mostrar el gráfico
    fig.show()

"""**Definición:**  
La función `graficar_correlacion` genera un gráfico de dispersión utilizando la librería Plotly Express para visualizar la correlación entre dos variables en un DataFrame, incluyendo una línea de regresión.

**Parámetros:**
- `df`: El DataFrame que contiene los datos.
- `variable_x`: El nombre de la columna que se representará en el eje x.
- `variable_y`: El nombre de la columna que se representará en el eje y.

**Utilidad:**  
Esta función permite crear gráficos de dispersión que muestran la relación entre dos variables cuantitativas, facilitando la identificación de patrones y tendencias en los datos. La inclusión de una línea de regresión ayuda a visualizar la fuerza y dirección de la correlación entre las variables analizadas.

### graficar_proporciones
"""

def graficar_proporciones(df, variable_categorica_1, variable_categorica_2):
    # Crear el histograma con barras agrupadas por la segunda variable categórica
    fig = px.histogram(df, x=variable_categorica_1, color=variable_categorica_2,
                       title='Análisis de múltiples variables categóricas',
                       labels={variable_categorica_1: f'Categoría: {variable_categorica_1}',
                               variable_categorica_2: f'Grupo: {variable_categorica_2}'},
                       text_auto=True,
                       barmode='group')

    # Actualizar títulos del gráfico
    fig.update_layout(yaxis_title='Volumen',
                      legend_title=variable_categorica_2,
                      bargap=0.2)

    # Actualizar el fondo del gráfico a blanco
    fig.update_layout({
        'plot_bgcolor': 'rgba(255, 255, 255, 1)',
        'xaxis': {'showgrid': True, 'gridcolor': 'lightgrey'},
        'yaxis': {'showgrid': True, 'gridcolor': 'lightgrey'}
    })

    # Mostrar el gráfico
    fig.show()

"""**Definición:**  
La función `graficar_proporciones` genera un histograma agrupado utilizando la librería Plotly Express para visualizar las proporciones de una variable categórica en función de otra variable categórica en un DataFrame.

**Parámetros:**
- `df`: El DataFrame que contiene los datos.
- `variable_categorica_1`: El nombre de la primera columna categórica que se representará en el eje x.
- `variable_categorica_2`: El nombre de la segunda columna categórica que se utilizará para agrupar los datos en el gráfico.

**Utilidad:**  
Esta función permite crear histogramas que muestran la distribución de frecuencias de una variable categórica desglosada por otra variable categórica, facilitando el análisis comparativo entre grupos. Es útil para identificar patrones y relaciones en datos categóricos, mejorando la comprensión de las proporciones relativas entre las categorías analizadas.

### realizar_crosstab
"""

def realizar_crosstab(df, variable_categorica_1, variable_categorica_2, normalize):
    # Crear una tabla de contingencia con conteos absolutos
    aux_p1 = pd.crosstab(df[variable_categorica_1], df[variable_categorica_2], margins=True)

    # Crear una tabla de contingencia con porcentajes
    aux_p2 = (pd.crosstab(df[variable_categorica_1],
                          df[variable_categorica_2],
                          normalize=normalize,
                          margins=True
                          ) * 100).round(2)

    # Concatenar ambas tablas para tener conteos y porcentajes en una sola tabla
    tabla_contingencia = pd.concat([aux_p1, aux_p2], axis=1)

    return tabla_contingencia

"""**Definición:**  
La función `realizar_crosstab` genera una tabla de contingencia que muestra tanto los conteos absolutos como los porcentajes de dos variables categóricas en un DataFrame.

**Parámetros:**
- `df`: El DataFrame que contiene los datos.
- `variable_categorica_1`: El nombre de la primera columna categórica que se utilizará para la tabla de contingencia.
- `variable_categorica_2`: El nombre de la segunda columna categórica que se utilizará para la tabla de contingencia.
- `normalize`: Un booleano que indica si se deben calcular los porcentajes en la tabla (True) o solo los conteos absolutos (False).

**Utilidad:**  
Esta función permite crear una tabla de contingencia que facilita el análisis de la relación entre dos variables categóricas, mostrando tanto los conteos como los porcentajes de las combinaciones de categorías. Es útil para identificar patrones, asociaciones y la distribución relativa entre diferentes grupos en el conjunto de datos.

### realizar_correlaciones
"""

def realizar_correlaciones(df, listado_variables):
    sns.heatmap(df[listado_variables].corr(), annot = True)

"""**Definición:**
La función `realizar_correlaciones` genera un mapa de calor que muestra la correlación entre un conjunto de variables en un DataFrame.

**Parámetros:**
- `df`: El DataFrame con los datos a analizar.
- `listado_variables`: Una lista de nombres de columnas del DataFrame que se utilizarán para calcular y visualizar las correlaciones.

**Utilidad:**
Es útil para identificar relaciones y patrones entre las variables seleccionadas, lo que permite interpretar cómo una variable puede influir en otra dentro del conjunto de datos.
"""
