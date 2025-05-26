from flask import Flask, render_template, request
import pandas as pd
import folium
import os
from folium.plugins import MarkerCluster

app = Flask(__name__)

# Ruta al CSV
DATA_PATH = 'datos/consumo_agua_historico_2019.csv'

# Cargar y limpiar el DataFrame
def cargar_datos():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.lower().str.strip()
    return df

# Generar el mapa con Folium
def generar_mapa(df_filtrado):
    m = folium.Map(location=[19.4326, -99.1332], zoom_start=11)
    cluster = MarkerCluster().add_to(m)

    for _, row in df_filtrado.iterrows():
        if pd.notna(row['latitud']) and pd.notna(row['longitud']):
            popup = folium.Popup(
                f"<strong>Alcaldía:</strong> {row['alcaldia']}<br>"
                f"<strong>Consumo total:</strong> {row['consumo_total_mixto']} m³",
                max_width=250
            )
            folium.CircleMarker(
                location=[row['latitud'], row['longitud']],
                radius=5,
                color='blue',
                fill=True,
                fill_opacity=0.6,
                popup=popup
            ).add_to(cluster)

    mapa_path = os.path.join('static', 'mapa.html')
    m.save(mapa_path)
    return mapa_path

# Ruta principal
@app.route('/', methods=['GET', 'POST'])
def index():
    df = cargar_datos()
    alcaldias = sorted(df['alcaldia'].dropna().unique())
    seleccionadas = request.form.getlist('alcaldias')

    if seleccionadas:
        df = df[df['alcaldia'].isin(seleccionadas)]

    generar_mapa(df)

    return render_template('index.html', alcaldias=alcaldias, seleccionadas=seleccionadas)

if __name__ == '__main__':
    app.run(debug=True, port=5001)