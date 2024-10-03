import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import statsmodels.api as sm
import google.generativeai as genai  # Importar la biblioteca de GEMINI

# Tu clave API de GEMINI
api_key = "AIzaSyDdUqWeUe_Oer7dtyK1q_zgv8pEuSbDOhM"

# Configurar la API de GEMINI utilizando la clave API directamente
genai.configure(api_key=api_key)

# Configuración de estilo para Seaborn
sns.set_style("whitegrid")

# Título de la aplicación
st.title("Analista de Datos")

# Cargar el archivo de Excel
uploaded_file = st.file_uploader("Cargar archivo Excel", type=["xlsx", "xls"])

if uploaded_file:
    # Leer el archivo Excel
    df = pd.read_excel(uploaded_file)
    st.subheader("Datos Cargados")
    st.dataframe(df)

    # Crear pestañas para visualización y análisis
    tab1, tab2, tab3, tab4 = st.tabs(["Visualización", "Informe Combinado", "Limpieza de Datos", "Analizador de Datos IA"])

    with tab1:
        st.subheader("Datos Cargados")
        st.dataframe(df)

        st.subheader("Informe Automático por Fila")

        # Agregar botón para generar gráficos
        if st.button("Generar Gráficos"):
            # Iterar sobre cada fila para generar informes automáticos
            for index, row in df.iterrows():
                st.write(f"### Informe para la fila: {row.iloc[0]}")

                # Extraer los datos desde la segunda columna en adelante
                values = row.iloc[1:].values
                columns = row.index[1:]

                # Convertir a números, manejando errores para filas con valores no numéricos
                try:
                    values = pd.to_numeric(values, errors='coerce')
                except Exception as e:
                    st.write(f"Error al convertir valores en la fila {index}: {e}")
                    continue

                # Filtrar valores NaN
                if np.all(np.isnan(values)):
                    st.write(f"La fila {row.iloc[0]} no contiene datos numéricos válidos.")
                    continue

                fig, axs = plt.subplots(2, 2, figsize=(15, 10))

                # Gráfico de líneas
                axs[0, 0].plot(columns, values, marker='o')
                axs[0, 0].set_title(f"Líneas: {row.iloc[0]}")
                axs[0, 0].set_xlabel("Meses")
                axs[0, 0].set_ylabel("Valores")
                axs[0, 0].tick_params(axis='x', rotation=45)

                # Gráfico de barras
                axs[0, 1].bar(columns, values)
                axs[0, 1].set_title(f"Barras: {row.iloc[0]}")
                axs[0, 1].set_xlabel("Meses")
                axs[0, 1].set_ylabel("Valores")
                axs[0, 1].tick_params(axis='x', rotation=45)

                # Gráfico de área
                axs[1, 0].fill_between(columns, values, alpha=0.5)
                axs[1, 0].set_title(f"Área: {row.iloc[0]}")
                axs[1, 0].set_xlabel("Meses")
                axs[1, 0].set_ylabel("Valores")
                axs[1, 0].tick_params(axis='x', rotation=45)

                # Gráfico de dispersión
                axs[1, 1].scatter(columns, values)
                axs[1, 1].set_title(f"Dispersión: {row.iloc[0]}")
                axs[1, 1].set_xlabel("Meses")
                axs[1, 1].set_ylabel("Valores")
                axs[1, 1].tick_params(axis='x', rotation=45)

                st.pyplot(fig)
                st.write("---")

    with tab2:
        st.subheader("Informe Combinado")

        # Usar el índice como identificador y mostrar los valores de la primera columna
        df['Identificador'] = df.iloc[:, 0]
        selected_row_name = st.selectbox("Seleccione la fila para analizar", df['Identificador'])

        # Seleccionar columnas para el análisis
        selected_columns = st.multiselect("Seleccione las columnas para analizar", df.columns.drop('Identificador'))

        # Seleccionar el tipo de gráfico
        chart_type_combined = st.selectbox(
            "Seleccione el tipo de gráfico",
            [
                "Gráfico de Líneas",
                "Gráfico de Barras",
                "Gráfico de Dispersión",
                "Histograma",
                "Gráfico de Área",
                "Gráfico de Cajas",
                "Gráfico de Violín",
                "Gráfico Circular",
            ],
        )

        if selected_row_name and selected_columns:
            # Filtrar la fila seleccionada por su valor en la columna de identificación
            filtered_values = df[df['Identificador'] == selected_row_name][selected_columns].squeeze()

            # Mostrar los valores seleccionados
            st.write(f"**Valores seleccionados para la fila '{selected_row_name}':**")
            st.write(filtered_values)

            # Calcular estadísticas adicionales
            st.write(f"- **Promedio**: {filtered_values.mean()}")
            st.write(f"- **Mediana**: {filtered_values.median()}")
            st.write(f"- **Varianza**: {filtered_values.var()}")
            st.write(f"- **Desviación Estándar**: {filtered_values.std()}")
            st.write(f"- **Valor Máximo**: {filtered_values.max()}")
            st.write(f"- **Valor Mínimo**: {filtered_values.min()}")

            # Generar el gráfico de los valores seleccionados
            st.write("**Gráfico de los valores seleccionados:**")
            fig, ax = plt.subplots()

            if chart_type_combined == "Gráfico de Barras":
                filtered_values.plot(kind='bar', ax=ax)
            elif chart_type_combined == "Gráfico de Líneas":
                filtered_values.plot(kind='line', ax=ax)
                ax.set_xticks(range(len(filtered_values.index)))
                ax.set_xticklabels(filtered_values.index, rotation=45)
            elif chart_type_combined == "Gráfico de Dispersión":
                ax.scatter(filtered_values.index, filtered_values.values)
                ax.set_xticks(range(len(filtered_values.index)))
                ax.set_xticklabels(filtered_values.index, rotation=45)
            elif chart_type_combined == "Histograma":
                sns.histplot(filtered_values, kde=True, ax=ax)
            elif chart_type_combined == "Gráfico de Área":
                filtered_values.plot(kind='area', ax=ax)
            elif chart_type_combined == "Gráfico de Cajas":
                sns.boxplot(data=filtered_values, ax=ax)
            elif chart_type_combined == "Gráfico de Violín":
                sns.violinplot(data=filtered_values, ax=ax)
            elif chart_type_combined == "Gráfico Circular":
                filtered_values.plot(kind='pie', ax=ax, autopct='%1.1f%%')
                ax.set_ylabel('')

            ax.set_ylabel("Valor")
            ax.set_title(f"{selected_row_name} - {', '.join(selected_columns)}")
            st.pyplot(fig)

            # Funcionalidad adicional en Informe Combinado

            # Análisis de Correlación
            if st.checkbox("Mostrar Matriz de Correlación"):
                correlation_matrix = df[selected_columns].corr()
                st.write("### Matriz de Correlación")
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)

            # Análisis de Tendencias
            if st.checkbox("Mostrar Tendencia"):
                if len(selected_columns) >= 2:
                    fig, ax = plt.subplots()
                    sns.regplot(x=selected_columns[0], y=selected_columns[1], data=df, ax=ax)
                    st.pyplot(fig)
                else:
                    st.warning("Seleccione al menos dos columnas para mostrar la tendencia.")

            # Análisis de Distribución
            if st.checkbox("Mostrar Distribución"):
                for col in selected_columns:
                    st.write(f"### Distribución de {col}")
                    fig, ax = plt.subplots()
                    sns.histplot(df[col], kde=True, ax=ax)
                    st.pyplot(fig)

            # Análisis de Componentes Principales (PCA)
            if st.checkbox("Realizar PCA"):
                pca = PCA(n_components=2)
                principal_components = pca.fit_transform(df[selected_columns].dropna())
                pca_df = pd.DataFrame(data=principal_components, columns=['PCA1', 'PCA2'])
                st.write("### Análisis de Componentes Principales")
                fig, ax = plt.subplots()
                ax.scatter(pca_df['PCA1'], pca_df['PCA2'])
                ax.set_xlabel('PCA1')
                ax.set_ylabel('PCA2')
                ax.set_title('Gráfico de Componentes Principales')
                st.pyplot(fig)

            # Análisis de Series Temporales
            if st.checkbox("Análisis de Series Temporales"):
                date_column = st.selectbox("Seleccione la columna de fechas", df.columns)
                series_column = st.selectbox("Seleccione la columna de valores", selected_columns)
                ts_data = df.set_index(date_column)[series_column].dropna()

                decomposition = sm.tsa.seasonal_decompose(ts_data, model='additive', period=12)
                fig = decomposition.plot()
                st.pyplot(fig)

            # Análisis de Clustering (K-means)
            if st.checkbox("Realizar Clustering"):
                k = st.slider("Seleccione el número de clusters", 2, 10, 3)
                kmeans = KMeans(n_clusters=k)
                clusters = kmeans.fit_predict(df[selected_columns].dropna())
                pca = PCA(n_components=2)
                principal_components = pca.fit_transform(df[selected_columns].dropna())
                pca_df = pd.DataFrame(data=principal_components, columns=['PCA1', 'PCA2'])
                pca_df['Cluster'] = clusters

                st.write("### Clustering (K-means)")
                fig, ax = plt.subplots()
                sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=pca_df, palette='viridis', ax=ax)
                st.pyplot(fig)

            # Análisis de Valores Atípicos
            if st.checkbox("Mostrar Valores Atípicos"):
                for col in selected_columns:
                    st.write(f"### Valores Atípicos en {col}")
                    fig, ax = plt.subplots()
                    sns.boxplot(x=df[col], ax=ax)
                    st.pyplot(fig)

    with tab3:
        st.subheader("Limpieza de Datos")

        st.write("Seleccione las filas y columnas que desea conservar y luego descargue el archivo filtrado.")

        # Selección de columnas
        selected_columns_clean = st.multiselect("Seleccione las columnas que desea conservar", df.columns)

        # Selección de filas
        if selected_columns_clean:
            selected_rows_clean = st.multiselect("Seleccione las filas que desea conservar", df.iloc[:, 0].unique())

            if selected_rows_clean:
                # Filtrar los datos
                df_filtered = df[df.iloc[:, 0].isin(selected_rows_clean)][selected_columns_clean]

                # Mostrar la tabla filtrada
                st.write("### Tabla Filtrada:")
                st.dataframe(df_filtered)

                # Descargar el archivo filtrado
                csv = df_filtered.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Descargar archivo filtrado",
                    data=csv,
                    file_name='archivo_filtrado.csv',
                    mime='text/csv',
                )
            else:
                st.warning("Seleccione al menos una fila para continuar.")
        else:
            st.warning("Seleccione al menos una columna para continuar.")

    with tab4:
        st.subheader("Análisis de Datos Automático con IA")

        if st.button("Analizar"):
            # Preparar los datos para ser enviados a la IA
            df_as_text = df.to_csv(index=False)

            # Crear el prompt personalizado
            input_text = f"""
Eres un analista de datos experto con más de 10 años de experiencia. Tu tarea es realizar un análisis completo y profundo de la siguiente base de datos, identificando patrones ocultos y generando insights valiosos. Por favor, sigue estos pasos detalladamente:

1. **Planteamiento del Problema:**
   - Identifica posibles problemas o áreas de interés que pueden ser exploradas con estos datos.
   - Plantea preguntas clave que el análisis debería responder.

2. **Recolección y Exploración de Datos:**
   - Describe las características generales de los datos (número de registros, variables, tipos de datos).
   - Detecta valores faltantes, datos atípicos y cualquier inconsistencia en los datos.

3. **Preparación y Limpieza de Datos:**
   - Propón métodos para manejar valores faltantes y outliers.
   - Sugiere transformaciones o normalizaciones necesarias para preparar los datos para el análisis.

4. **Análisis Exploratorio de Datos (EDA):**
   - Realiza un análisis detallado buscando patrones, tendencias, correlaciones y relaciones significativas entre las variables.
   - Identifica patrones o relaciones que no son evidentes a simple vista.

5. **Generación de Hipótesis y Preguntas de Investigación:**
   - Formula hipótesis basadas en los patrones encontrados.
   - Propón posibles temas de investigación que podrían ser explorados más a fondo.

6. **Modelado y Pruebas (si aplica):**
   - Sugiere modelos estadísticos o algoritmos de machine learning que podrían aplicarse a los datos.
   - Explica cómo estos modelos podrían ayudar a extraer más información de los datos.

7. **Interpretación y Comunicación de Resultados:**
   - Interpreta los hallazgos del análisis, explicando su significado en el contexto del problema.
   - Sugiere visualizaciones que podrían ayudar a comunicar los resultados de manera efectiva a una audiencia no técnica.

8. **Recomendaciones y Próximos Pasos:**
   - Propón oportunidades adicionales para profundizar en el análisis.
   - Indica qué otros indicadores o variables podrían incluirse en futuros análisis para mejorar la comprensión del problema.
   - Plantea preguntas adicionales que podrían guiar investigaciones futuras.

9. **Creatividad y Pensamiento Crítico:**
   - Utiliza tu experiencia para aportar insights que puedan no ser obvios.
   - Sé creativo en tus sugerencias y recomendaciones.

10. **Comunicación y Visualización**: Sugiere qué tipos de visualizaciones podrían ayudar a comunicar mejor los hallazgos a una audiencia no técnica.

11. **Propuestas y Recomendaciones**:
   - **Oportunidades de Análisis**: Propón oportunidades adicionales para profundizar en el análisis de los datos.
   - **Temas de Investigación**: Sugiere posibles temas de investigación que podrían explorarse a partir de los datos.
   - **Indicadores Adicionales**: Indica qué otros indicadores o variables podrían incluirse en futuros análisis para complementar y mejorar la comprensión del problema.
12. **Generador de Tesis e investigaciones**
    - Menciona que investigaciones o temas de tesis podrian generarse apartir de la base de datos.
    -Haz una propuesta de investigacion generando titulos de tesis o investigaciones a raiz de las hipotesis planteadas



Base de datos:
{df_as_text}
"""

            # Seleccionar el modelo de GEMINI
            model = genai.GenerativeModel("gemini-1.5-flash")

            # Generar el contenido a partir del input
            response = model.generate_content(input_text)

             # Mostrar el resultado generado
            if response and response.candidates:
                st.success("Análisis completado exitosamente")
                generated_content = response.candidates[0].content.parts[0].text
                st.write("### Resultado del Análisis:")
                st.write(generated_content)

                # Agregar botón para descargar el análisis en un archivo de texto
                st.download_button(
                    label="Descargar Análisis en TXT",
                    data=generated_content,
                    file_name='analisis_IA.txt',
                    mime='text/plain',
                )
            else:
                st.error("No se obtuvo una respuesta válida de la API.")
else:
    st.warning("Por favor, sube una base de datos primero.")