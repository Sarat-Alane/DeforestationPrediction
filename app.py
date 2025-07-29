import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
import geopandas as gpd
import datetime
import numpy as np
import joblib
import ee
from shapely.geometry import shape
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Initialize EE
ee.Initialize(project='ringed-trail-454308-d2')

# Load model and scaler
model = joblib.load("this_ensemble_model.pkl")
scaler = joblib.load("this_scaler.pkl")

# Load grid
@st.cache_data
def load_grid():
    return gpd.read_file("Maharashtra_5x5km_Grid.geojson")

gdf = load_grid()

# Streamlit setup
st.set_page_config(layout="wide")
st.title("🌳 Deforestation Analysis Tool")

# Navigation menu
page = st.sidebar.radio("📂 Navigation", [
    "Region Selection",
    "Predict Forest Cover",
    "Timeline of Tree Cover",
    "Environmental Statistics"
])

# Page 0: Region Selection
if page == "Region Selection":
    st.header(":pencil2: Draw a Region to Analyze")
    m = folium.Map(location=[19.5, 74], zoom_start=7.5)
    Draw(export=True).add_to(m)
    map_data = st_folium(m, height=600, width=900)

    if map_data.get("last_active_drawing"):
        st.session_state["last_active_drawing"] = map_data["last_active_drawing"]
        st.success("✅ Region Selected! Now switch pages from the left sidebar.")

    st.stop()

# Reset button
if st.sidebar.button("🔄 Reset Region"):
    st.session_state.pop("last_active_drawing", None)
    st.rerun()

if "last_active_drawing" not in st.session_state:
    st.warning("⚠️ Please go to 'Region Selection' and draw a region first.")
    st.stop()

# Selected region
drawn_shape = shape(st.session_state["last_active_drawing"]["geometry"])
selected_gdf = gdf[gdf.intersects(drawn_shape)].copy()
region = ee.Geometry.Polygon([list(drawn_shape.exterior.coords)])

if selected_gdf.empty:
    st.warning("⚠️ No grid found in selected region.")
    st.stop()

# ------------------ Helper: Compute EE features ------------------
def compute_features(lon, lat, start_date, end_date):
    point = ee.Geometry.Point([lon, lat])
    try:
        mod13 = ee.ImageCollection("MODIS/061/MOD13Q1") \
            .filterDate(str(start_date), str(end_date)) \
            .filterBounds(point) \
            .select(["NDVI", "EVI"]).mean()

        evi = mod13.select("EVI").multiply(0.0001).reduceRegion(ee.Reducer.mean(), point, 250).get("EVI").getInfo()
        ndvi = mod13.select("NDVI").multiply(0.0001).reduceRegion(ee.Reducer.mean(), point, 250).get("NDVI").getInfo()

        def add_ndmi(img):
            return img.addBands(img.normalizedDifference(['sur_refl_b02', 'sur_refl_b06']).rename("NDMI"))

        ndmi_img = ee.ImageCollection("MODIS/061/MOD09GA") \
            .filterBounds(point).filterDate(str(start_date), str(end_date)) \
            .map(add_ndmi).select("NDMI").mean()
        ndmi = ndmi_img.reduceRegion(ee.Reducer.mean(), point, 500).get("NDMI").getInfo()

        precip = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
            .filterDate(str(start_date), str(end_date)) \
            .filterBounds(point).sum().select("precipitation") \
            .reduceRegion(ee.Reducer.mean(), point, 5000).get("precipitation").getInfo()

        lst_img = ee.ImageCollection("MODIS/061/MOD11A1") \
            .filterDate(str(start_date), str(end_date)) \
            .filterBounds(point).select("LST_Day_1km").mean()
        lst = lst_img.multiply(0.02).subtract(273.15).reduceRegion(ee.Reducer.mean(), point, 1000).get("LST_Day_1km").getInfo()

        tree_img = ee.ImageCollection("MODIS/061/MOD44B") \
            .filterBounds(point).select("Percent_Tree_Cover").mean()
        treecover = tree_img.reduceRegion(ee.Reducer.mean(), point, 250).get("Percent_Tree_Cover").getInfo()

        return [evi, ndvi, ndmi, lst, precip, treecover]
    except:
        return [None]*6

# ------------------ Page 1 ------------------
if page == "Predict Forest Cover":
    st.header("🧐 Forest Cover Prediction")
    st.info(f"Selected {len(selected_gdf)} grid cells for prediction.")

    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=30)

    features = []
    for _, row in selected_gdf.iterrows():
        lon, lat = row.geometry.centroid.x, row.geometry.centroid.y
        feats = compute_features(lon, lat, start_date, end_date)
        features.append((row["geometry"], feats))

    X = [[f[1][i] for i in [0, 2, 3, 4, 5]] if None not in f[1] else [0]*5 for f in features]
    X_scaled = scaler.transform(np.array(X))
    y_pred = model.predict(X_scaled)

    param_data = [{
        "EVI": round(f[1][0], 4) if f[1][0] is not None else "N/A",
        "NDMI": round(f[1][2], 4) if f[1][2] is not None else "N/A",
        "LST": round(f[1][3], 2) if f[1][3] is not None else "N/A",
        "Precip": round(f[1][4], 2) if f[1][4] is not None else "N/A",
        "Tree Cover": round(f[1][5], 1) if f[1][5] is not None else "N/A"
    } for f in features]

    result_map = folium.Map(location=[region.centroid().coordinates().getInfo()[1],
                                      region.centroid().coordinates().getInfo()[0]], zoom_start=9.5)

    for idx, (geom, pred) in enumerate(zip([f[0] for f in features], y_pred)):
        tooltip_text = "<br>".join([f"{k}: {v}" for k, v in param_data[idx].items()])
        tooltip_text += "<br><b>Prediction</b>: " + ("Forest" if pred == 0 else "Deforested")
        fill = "green" if pred == 0 else "red"

        folium.GeoJson(
            geom,
            tooltip=tooltip_text,
            style_function=lambda x, fill_color=fill: {
                "fillColor": fill_color,
                "color": "black",
                "weight": 0.5,
                "fillOpacity": 0.6
            }
        ).add_to(result_map)

    st.subheader("🗺️ Predicted Forest Cover")
    st_folium(result_map, height=600, width=900)

# ------------------ Page 2 ------------------
if page == "Timeline of Tree Cover":
    st.header("📽️ Animated Timeline of EVI (2016–2025)")
    years = list(range(2016, 2026))

    def create_evi_image(year):
        evi = ee.ImageCollection("MODIS/061/MOD13Q1") \
            .filterDate(f"{year}-01-01", f"{year}-12-31") \
            .filterBounds(region) \
            .select("EVI") \
            .mean() \
            .multiply(0.0001) \
            .clip(region)

        return evi.visualize(min=0, max=0.8, palette=['#ffffcc', '#c2e699', '#78c679', '#31a354', '#006837']).set({'label': str(year)})

    image_list = [create_evi_image(y) for y in years]
    gif_collection = ee.ImageCollection(image_list)

    gif_url = gif_collection.getVideoThumbURL({
        'dimensions': 400,
        'region': region.bounds(),
        'framesPerSecond': 1,
        'format': 'gif'
    })

    st.subheader("🎮 EVI Animation Over Selected Region")
    st.image(gif_url, caption="🌿 Animated EVI Change (2016–2025)", width=400)

    st.markdown("### 📊 Yearly EVI Snapshots")
    thumbs = []
    for year in years:
        evi = ee.ImageCollection("MODIS/061/MOD13Q1") \
            .filterDate(f"{year}-01-01", f"{year}-12-31") \
            .filterBounds(region).select("EVI").mean() \
            .multiply(0.0001).clip(region)

        vis = {
            "min": 0,
            "max": 0.8,
            "palette": ['#ffffcc', '#c2e699', '#78c679', '#31a354', '#006837']
        }

        url = evi.visualize(**vis).getThumbURL({
            "region": region.bounds().getInfo(),
            "dimensions": 150,
            "format": "png"
        })
        thumbs.append((year, url))

    cols = st.columns(len(thumbs))
    for i, (yr, url) in enumerate(thumbs):
        with cols[i]:
            st.image(url, caption=str(yr), use_container_width=True)

# ------------------ Page 3 ------------------
if page == "Environmental Statistics":
    st.header("📊 Environmental Statistics")

    param_bands = {
        "NDVI": ("MODIS/061/MOD13Q1", "NDVI", 250, 0.0001),
        "EVI": ("MODIS/061/MOD13Q1", "EVI", 250, 0.0001),
        "Precipitation": ("UCSB-CHG/CHIRPS/DAILY", "precipitation", 5000, 1),
        "LST (°C)": ("MODIS/061/MOD11A1", "LST_Day_1km", 1000, 0.02),
        "Tree Cover (%)": ("MODIS/061/MOD44B", "Percent_Tree_Cover", 250, 1)
    }

    start_year = 2015
    end_year = 2023
    data = {k: [] for k in param_bands}
    labels = []

    for y in range(start_year, end_year + 1):
        start = f"{y}-01-01"
        end = f"{y}-12-31"
        labels.append(str(y))
        for name, (ic, band, scale, factor) in param_bands.items():
            coll = ee.ImageCollection(ic).filterDate(start, end).filterBounds(region)
            img = coll.select(band).mean()
            if name == "LST (°C)":
                img = img.multiply(factor).subtract(273.15)
            elif factor != 1:
                img = img.multiply(factor)
            val = img.reduceRegion(ee.Reducer.mean(), region, scale).get(band).getInfo()
            data[name].append(val if val is not None else 0)

    df = pd.DataFrame(data, index=labels)

    # ---- Graph 1: EVI vs Precipitation (Normalized) ----
    st.markdown("### 📈 Normalized EVI vs Precipitation (2015–2023)")
    norm_df = df[['EVI', 'Precipitation']].copy()
    norm_df = (norm_df - norm_df.min()) / (norm_df.max() - norm_df.min())

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(labels, norm_df['EVI'], marker='o', label="EVI")
    ax1.plot(labels, norm_df['Precipitation'], marker='s', label="Precipitation")
    ax1.set_ylabel("Normalized Value")
    ax1.set_xlabel("Year")
    ax1.set_title("EVI vs Precipitation")
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    # ---- Graph 2: Correlation Matrix ----
    st.markdown("### 📊 Correlation Matrix")
    corr = df.corr()
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    im = ax2.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax2.set_xticks(np.arange(len(corr.columns)))
    ax2.set_yticks(np.arange(len(corr.columns)))
    ax2.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax2.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax2.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")
    fig2.colorbar(im, ax=ax2)
    ax2.set_title("Correlation Matrix")
    st.pyplot(fig2)

    # ---- Graph 3: Monthly Trends (Normalized) ----
    st.markdown("### 📅 Monthly Trends of EVI and Precipitation")
    monthly_df = pd.DataFrame(columns=["Month", "EVI", "Precipitation"])

    for month in range(1, 13):
        start = f"2023-{month:02d}-01"
        end = f"2023-{month:02d}-28"
        evi = ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(start, end).filterBounds(region).select("EVI").mean().multiply(0.0001)
        precip = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(start, end).filterBounds(region).select("precipitation").sum()

        evi_val = evi.reduceRegion(ee.Reducer.mean(), region, 250).get("EVI").getInfo()
        precip_val = precip.reduceRegion(ee.Reducer.mean(), region, 5000).get("precipitation").getInfo()

        monthly_df.loc[month - 1] = [month, evi_val or 0, precip_val or 0]

    monthly_df["EVI_norm"] = (monthly_df["EVI"] - monthly_df["EVI"].min()) / (monthly_df["EVI"].max() - monthly_df["EVI"].min())
    monthly_df["Precip_norm"] = (monthly_df["Precipitation"] - monthly_df["Precipitation"].min()) / (monthly_df["Precipitation"].max() - monthly_df["Precipitation"].min())

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.plot(monthly_df["Month"], monthly_df["EVI_norm"], marker='o', label="EVI (Normalized)")
    ax3.plot(monthly_df["Month"], monthly_df["Precip_norm"], marker='s', label="Precipitation (Normalized)")
    ax3.set_xticks(range(1, 13))
    ax3.set_xlabel("Month")
    ax3.set_ylabel("Normalized Value")
    ax3.set_title("Monthly EVI & Precipitation (2023)")
    ax3.grid(True)
    ax3.legend()
    st.pyplot(fig3)

