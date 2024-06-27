import folium
import streamlit as st
from pymongo import MongoClient
import os
import pandas as pd
from dotenv import load_dotenv
import altair as alt
from streamlit_folium import st_folium
from streamlit_navigation_bar import st_navbar


MONGO_URI = os.environ.get("MONGO_URI")
APP_TITLE = 'Emotion Analysis Terhadap Kemenangan Paslon Prabowo-Gibran Pada Pemilihan Presiden 2024'


def get_dominant_emotion(row):
    emotions = ['bahagia', 'jijik', 'lainnya', 'marah', 'sedih', 'takut', 'terkejut']
    dominant_emotion = row[emotions].idxmax()
    return dominant_emotion


def get_emotion_color(emotion):
    color_dict = {
        'bahagia': '#FFD700',  # Gold
        'jijik': '#8A2BE2',  # BlueViolet
        'lainnya': '#A9A9A9',  # DarkGray
        'marah': '#FF4500',  # OrangeRed
        'sedih': '#1E90FF',  # DodgerBlue
        'takut': '#4B0082',  # Indigo
        'terkejut': '#FF69B4',  # HotPink
    }
    return color_dict.get(emotion, '#FFFFFF')  # Default to white if not found


# Database connection function
@st.cache_resource
def get_database(database_name, connection_str):
    # Creating the connection
    client = MongoClient(connection_str)
    # Accessing a database and returning it
    return client[database_name]


@st.cache_resource
def get_data():
    db_viz = get_database("local", MONGO_URI)
    tweet_collection_labeled = db_viz['data-tweet-election-2024']
    return pd.DataFrame(list(tweet_collection_labeled.find()))


def tweet_trends(df):
    df_date_index = df.copy()
    df_date_index['created_at'] = pd.to_datetime(df_date_index['created_at'])
    df_date_index.set_index('created_at', inplace=True)

    # Resample data to get the count of tweets per day
    tweets_per_day = df_date_index['full_text'].resample('D').count().reset_index()
    chart = alt.Chart(tweets_per_day).mark_line(point=True).encode(
        x=alt.X('created_at:T', title='Tanggal'),
        y=alt.Y('full_text:Q', title='Jumlah Tweet'),
        tooltip=[
            alt.Tooltip('created_at:T', title='Tanggal'),
            alt.Tooltip('full_text:Q', title='Jumlah Tweet')
        ]
    ).properties(
        title='Jumlah Tweet Dari Waktu ke Waktu',
        width=800,
        height=400
    ).configure_axis(
        labelAngle=45
    ).configure_title(
        anchor='start'
    ).interactive()

    return chart


@st.cache_resource(experimental_allow_widgets=True)
def display_map(df):
    df['dominant_emotion'] = df.apply(get_dominant_emotion, axis=1)
    df['color'] = df['dominant_emotion'].apply(get_emotion_color)
    map = folium.Map(location=[-3.46955730306146, 118.69628906250001],
                     zoom_start=5,
                     tiles="CartoDB Positron")
    choropleth = folium.Choropleth(
        geo_data='indonesia-prov.geojson',
        data=df,
        columns=['location', 'location_count'],
        key_on='feature.properties.Propinsi',
        line_opacity=0.9,
        highlight=True,
        fill_opacity=1,
        legend_name="Emotion",
        fill_color="YlOrRd",
        show=True,
    )
    choropleth.geojson.add_to(map)

    df_indexed = df.set_index('location')
    for feature in choropleth.geojson.data['features']:
        province_name = feature['properties']['Propinsi']
        feature['properties']['location_count'] = 'location_count: ' + '{:,}'.format(
            df_indexed.loc[province_name, 'location_count']) if province_name in list(
            df_indexed.index) else 'location_count: 0'
        feature['properties']['bahagia'] = 'bahagia: ' + '{:,}'.format(
            df_indexed.loc[province_name, 'bahagia']) if province_name in list(df_indexed.index) else 'bahagia: 0'
        feature['properties']['jijik'] = 'jijik: ' + '{:,}'.format(
            df_indexed.loc[province_name, 'jijik']) if province_name in list(df_indexed.index) else 'jijik: 0'
        feature['properties']['lainnya'] = 'lainnya: ' + '{:,}'.format(
            df_indexed.loc[province_name, 'lainnya']) if province_name in list(df_indexed.index) else 'lainnya: 0'
        feature['properties']['marah'] = 'marah: ' + '{:,}'.format(
            df_indexed.loc[province_name, 'marah']) if province_name in list(df_indexed.index) else 'marah: 0'
        feature['properties']['sedih'] = 'sedih: ' + '{:,}'.format(
            df_indexed.loc[province_name, 'sedih']) if province_name in list(df_indexed.index) else 'sedih: 0'
        feature['properties']['takut'] = 'takut: ' + '{:,}'.format(
            df_indexed.loc[province_name, 'takut']) if province_name in list(df_indexed.index) else 'takut: 0'
        feature['properties']['terkejut'] = 'terkejut: ' + '{:,}'.format(
            df_indexed.loc[province_name, 'terkejut']) if province_name in list(df_indexed.index) else 'terkejut: 0'

        if province_name in df_indexed.index:
            feature['properties']['dominant_emotion'] = df_indexed.loc[province_name, 'dominant_emotion']
            feature['properties']['color'] = df_indexed.loc[province_name, 'color']
        else:
            feature['properties']['dominant_emotion'] = 'None'
            feature['properties']['color'] = '#FFFFFF'

        # Add color to each feature
        color = feature['properties']['color']
        feature['style'] = {
            'fillColor': color,
            'color': color,
            'weight': 1,
            'fillOpacity': 0.2,
        }

    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(
            ['Propinsi', 'location_count', 'bahagia', 'jijik', 'lainnya', 'marah', 'sedih', 'takut', 'terkejut',
             'dominant_emotion'],
            labels=False
        )
    )
    st_map = st_folium(map, width=1020, height=500)

    province_name = ''
    if st_map['last_active_drawing']:
        province_name = st_map['last_active_drawing']['properties']['Propinsi']
    return st_map


def emotion_distribusion(df):
    # Count the values of the 'label' column
    df = df.copy()
    label_counts = df['label'].value_counts().reset_index()
    label_counts.columns = ['label', 'count']
    # Create a horizontal bar chart using Altair
    chart = alt.Chart(label_counts).mark_bar().encode(
        x=alt.X('count', title="Jumlah"),
        y=alt.Y('label', sort='-x', title="Emosi"),
        color=alt.Color('label', title='Emosi')
    ).properties(
        title='Distribusi Emosi '
    )
    return chart


def map_data_manipulation(df):
    df = df.copy()

    # Group by 'location' and 'label' and count the occurrences
    grouped = df.groupby(['location', 'label']).size().reset_index(name='count')

    # Pivot the table to get the desired format
    pivot_table = grouped.pivot(index='location', columns='label', values='count').fillna(0)

    # Add the total count for each location
    pivot_table['location_count'] = pivot_table.sum(axis=1)

    # Reorder columns to have 'location', 'location_count', followed by label counts
    pivot_table = pivot_table.reset_index()
    columns = ['location', 'location_count'] + [col for col in pivot_table.columns if
                                                col not in ['location', 'location_count']]
    pivot_table = pivot_table[columns]

    # Rename columns to be more descriptive if necessary
    pivot_table.columns = ['location', 'location_count'] + [str(col) for col in pivot_table.columns if
                                                            col not in ['location', 'location_count']]

    return pivot_table


def main():
    styles = {
        "nav": {
            "background-color": "white",
            "display": "flex",

            "height": ".01rem"
        },
    }
    options = {
        'show_menu': False
    }
    pages = ['Home']
    load_dotenv()
    st.set_page_config(page_title=APP_TITLE, layout="wide",page_icon="img.png")
    page = st_navbar(pages,
                     styles=styles,
                     options=options
                     )
    st.markdown(f"<h1 style='text-align: center;'>{APP_TITLE}</h1>", unsafe_allow_html=True)
    df = get_data()

    tweet_trends_chart = tweet_trends(df)
    emotion_distribusion_chart = emotion_distribusion(df)

    col1, col2 = st.columns([3, 2])
    map_data = map_data_manipulation(df)

    with col1:
        map = display_map(map_data)
        st.write(df)
    with col2:
        st.altair_chart(emotion_distribusion_chart, use_container_width=True)
        st.altair_chart(tweet_trends_chart, use_container_width=True)


if __name__ == "__main__":
    main()
