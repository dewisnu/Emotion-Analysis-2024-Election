import datetime

import folium
import streamlit as st
from matplotlib import pyplot as plt
from pymongo import MongoClient
import os
import pandas as pd
from dotenv import load_dotenv
import altair as alt
from streamlit_folium import folium_static
from streamlit_navigation_bar import st_navbar
from wordcloud import WordCloud
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

responsive_css = """
    background-color: white;
    color: black;
    font-family: Arial, sans-serif;
    font-size: 12px;
    padding: 10px;
    width: 90%; 
    max-width: 300px;
    height: auto;
    box-sizing: border-box;

    @media (max-width: 600px) {
        font-size: 10px;
        padding: 5px;
    }

    @media (min-width: 601px) and (max-width: 1024px) {
        font-size: 12px;
        padding: 10px;
    }

    @media (min-width: 1025px) {
        font-size: 14px;
        padding: 15px;
    }
"""


pretrained = "arthd24/indobert_emotion_base_V2"

model = AutoModelForSequenceClassification.from_pretrained(pretrained)
tokenizer = AutoTokenizer.from_pretrained(pretrained)
sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

MONGO_URI = os.environ.get("MONGO_URI")
APP_TITLE = 'Emotion Analysis Terhadap Kemenangan Paslon Prabowo-Gibran Pada Pemilihan Presiden 2024'


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
    tweet_collection_labeled = db_viz['data-tweet-election-2024-v2']
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
    map = folium.Map(location=[-3.46955730306146, 118.69628906250001],
                     zoom_start=5,
                     tiles="CartoDB Positron")
    choropleth = folium.Choropleth(
        geo_data='indonesia-prov.geojson',
        data=df,
        columns=['location', 'location_count'],
        key_on='feature.properties.Propinsi',
        line_opacity=0.9,
        line_color="Green",
        line_weight=0.9,
        fill_opacity=1,
        highlight=True,
        legend_name="Jumlah Tweet",
        fill_color="YlOrRd",
    )
    choropleth.add_to(map)

    df_indexed = df.set_index('location')
    for feature in choropleth.geojson.data['features']:
        province_name = feature['properties']['Propinsi']

        # Default values if province_name is not in df_indexed
        location_count = 'location_count: 0'
        neutral = 'Neutral: 0'
        anger = 'Anger: 0'
        joy = 'Joy: 0'
        fear = 'Fear: 0'
        sad = 'Sad: 0'
        love = 'Love: 0'
        try:
            if province_name in df_indexed.index:
                location_count = 'location_count: ' + '{:,}'.format(df_indexed.loc[province_name, 'location_count'])
                neutral = 'neutral: ' + '{:,}'.format(df_indexed.loc[province_name, 'Neutral'])
                anger = 'anger: ' + '{:,}'.format(df_indexed.loc[province_name, 'Anger'])
                joy = 'joy: ' + '{:,}'.format(df_indexed.loc[province_name, 'Joy'])
                fear = 'fear: ' + '{:,}'.format(df_indexed.loc[province_name, 'Fear'])
                sad = 'sad: ' + '{:,}'.format(df_indexed.loc[province_name, 'Sad'])
                love = 'love: ' + '{:,}'.format(df_indexed.loc[province_name, 'Love'])
        except KeyError:
            pass

        feature['properties']['location_count'] = location_count
        feature['properties']['neutral'] = neutral
        feature['properties']['anger'] = anger
        feature['properties']['joy'] = joy
        feature['properties']['fear'] = fear
        feature['properties']['sad'] = sad
        feature['properties']['love'] = love

    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(
            ['Propinsi', 'location_count', 'neutral', 'anger', 'joy', 'fear', 'love', ],
            labels=False,
            style=("background-color: white; color: black; font-family: arial; font-size: 20px; padding: 10px; ")
        )
    )

    folium_static(map, width=1450, height=700)


def emotion_distribusion(df):
    df = df.copy()
    label_counts = df['label'].value_counts().reset_index()
    label_counts.columns = ['label', 'count']

    chart = alt.Chart(label_counts).mark_bar().encode(
        x=alt.X('count', title="Jumlah"),
        y=alt.Y('label', sort='-x', title="Emosi"),
        color=alt.Color('label', title='Emosi')
    ).properties(
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


def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    st.pyplot(plt)


def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    st.pyplot(plt)




def main():

    load_dotenv()
    st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="img.png", initial_sidebar_state="expanded", )
    df = get_data()

    # Get the current date
    current_date = datetime.date.today()

    # Get the current year
    current_year = current_date.year

    # Get the first date of the current year
    first_date_of_year = datetime.date(current_year, 1, 1)

    # Get the first date of the first month of the current year
    first_date_of_first_month = datetime.date(current_year, 1, 1)

    st.sidebar.header('Pilih Rentang Tanggal Data')
    min_date = pd.to_datetime(df['created_at']).min().date()
    max_date = pd.to_datetime(df['created_at']).max().date()
    start_date = st.sidebar.date_input("Tanggal Awal", min_value=min_date, max_value=max_date, value=first_date_of_first_month)
    end_date = st.sidebar.date_input("Tanggal Akhir", min_value=min_date, max_value=max_date, value=max_date)

    filtered_df = df[(pd.to_datetime(df['created_at']).dt.date >= start_date) &
                     (pd.to_datetime(df['created_at']).dt.date <= end_date)]

    tweet_trends_chart = tweet_trends(filtered_df)
    emotion_distribusion_chart = emotion_distribusion(filtered_df)

    col1, col2 = st.columns([2, 1], gap="medium")
    map_data = map_data_manipulation(filtered_df)

    with col1:
        st.subheader("Peta Distribusi Jumlah Tweet di Indonesia Berdasarkan Provinsi")
        display_map(map_data)
        st.subheader("Jumlah Tweet Dari Waktu ke Waktu")
        st.altair_chart(tweet_trends_chart, use_container_width=True)
    with col2:
        st.subheader("Distribusi Jumlah Emosi dari tweet")
        st.altair_chart(emotion_distribusion_chart, use_container_width=True)
        st.subheader("Word Cloud of Tweets")
        all_text = " ".join(tweet for tweet in filtered_df['clean_text'])
        generate_wordcloud(all_text)
        filtered_df['word_count'] = filtered_df['clean_text'].apply(lambda x: len(x.split()))
        word_count_chart = alt.Chart(filtered_df).mark_bar().encode(
            alt.X('word_count', bin=alt.Bin(maxbins=30), title='Jumlah Kata per Tweet'),
            alt.Y('count()', title='Frekuensi')
        ).properties(
            width=600,
            height=400
        )
        st.altair_chart(word_count_chart, use_container_width=True)


if __name__ == "__main__":
    main()
