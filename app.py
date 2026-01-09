# -*- coding: utf-8 -*-
"""News_Streaming.ipynb - UPDATED WITH TRAIN/TEST SPLIT"""

import streamlit as st
import pandas as pd
import requests
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import altair as alt
import time

# --- INITIAL SETUP ---
st.set_page_config(
    page_title="PySpark News Sentiment Dashboard",
    page_icon="📰",
    layout="wide"
)

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# --- SPARK SESSION ---
@st.cache_resource
def get_spark():
    return SparkSession.builder \
        .appName("NewsSentimentDashboard") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()

spark = get_spark()

# --- NEWS API & DATA LABELING ---
try:
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
except KeyError:
    st.error("NEWS_API_KEY not found in secrets. Please set it in your Streamlit Cloud app settings.")
    st.stop()

def fetch_news(topic, page_size=20):  # Increased default for split
    """Fetches news articles and returns a Pandas DataFrame."""
    url = f"https://newsapi.org/v2/everything?q={topic}&pageSize={page_size}&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    try:
        r = requests.get(url)
        r.raise_for_status()
        articles = r.json().get("articles", [])
        valid_articles = [a for a in articles if a.get("title")]
        return pd.DataFrame([{"title": a["title"]} for a in valid_articles])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching news: {e}")
        return None

def label_sentiment_vader(df_pd):
    """Labels a DataFrame with sentiment using VADER."""
    sia = SentimentIntensityAnalyzer()
    def classify_title(title):
        score = sia.polarity_scores(title)["compound"]
        if score > 0.05:
            return "Positive"
        elif score < -0.05:
            return "Negative"
        else:
            return "Neutral"
    df_pd["sentiment"] = df_pd["title"].apply(classify_title)
    label_map = {"Positive": 2.0, "Neutral": 1.0, "Negative": 0.0}
    df_pd["label"] = df_pd["sentiment"].map(label_map)
    return df_pd

# --- UI COMPONENTS ---
st.title("📰 PySpark News Sentiment Dashboard")
st.markdown("""
This dashboard fetches live news, **trains with train/test split**, and predicts sentiment.
**Key improvement**: Proper validation accuracy prevents overfitting!
""")

with st.sidebar:
    st.header("⚙️ Controls")
    topic = st.text_input("Enter a News Topic", "finance")
    mode = st.radio("Select Mode", ["Bootstrap", "Predict"])
    num_articles = st.slider("Number of Articles (min 10 for split)", 10, 50, 20)
    train_ratio = st.slider("Train Ratio", 0.6, 0.9, 0.8, 0.05)
    run_button = st.button("Run Analysis")

# --- MAIN LOGIC ---
if run_button:
    with st.spinner(f"Fetching {num_articles} articles about '{topic}'..."):
        df_pd = fetch_news(topic, num_articles)

    if df_pd is None or df_pd.empty or len(df_pd) < 10:
        st.warning("Need at least 10 articles. Try different topic or increase count.")
    else:
        if mode == "Bootstrap":
            st.header("🛠️ Bootstrap Mode: Training with Train/Test Split")
            with st.spinner("Labeling, splitting, and training..."):
                df_pd_labeled = label_sentiment_vader(df_pd)
                df_spark = spark.createDataFrame(df_pd_labeled)
                
                # **FIX 1: Train/Test Split**
                train_df, test_df = df_spark.randomSplit([train_ratio, 1-train_ratio], seed=42)
                
                # Pipeline
                tokenizer = Tokenizer(inputCol="title", outputCol="words")
                hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=2000)
                idf = IDF(inputCol="rawFeatures", outputCol="features")
                lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10, regParam=0.001)
                pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, lr])
                
                # **FIX 2: Train on train_df only**
                model = pipeline.fit(train_df)
                
                # **FIX 3: Test on test_df only**
                train_predictions = model.transform(train_df)
                test_predictions = model.transform(test_df)
                
                evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
                train_accuracy = evaluator.evaluate(train_predictions)
                test_accuracy = evaluator.evaluate(test_predictions)  # **New: True test score**
                
                st.session_state['sentiment_model'] = model
                st.success("✅ Model trained with proper validation!")
                
                # **Display both metrics**
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Train Accuracy", f"{train_accuracy:.2%}")
                with col2:
                    st.metric("Test Accuracy", f"{test_accuracy:.2%}", delta=f"{test_accuracy-train_accuracy:+.1%}")
                
                st.info(f"✅ Test accuracy ({test_accuracy:.1%}) validates generalization!")
            
            st.subheader("Training Data Distribution")
            st.dataframe(df_pd_labeled[["title", "sentiment", "label"]].head(10), use_container_width=True)

        elif mode == "Predict":
            st.header("🔮 Predict Mode: New Articles")
            if 'sentiment_model' not in st.session_state:
                st.error("Run Bootstrap first to train model!")
            else:
                with st.spinner("Predicting on new data..."):
                    model = st.session_state['sentiment_model']
                    df_spark_new = spark.createDataFrame(df_pd)
                    predictions = model.transform(df_spark_new)
                    predictions_df = predictions.select("title", "prediction").toPandas()
                    label_map_reverse = {2.0: "Positive", 1.0: "Neutral", 0.0: "Negative"}
                    predictions_df["sentiment"] = predictions_df["prediction"].map(label_map_reverse)
                
                st.subheader("Predictions on Fresh Articles")
                st.dataframe(predictions_df[["title", "sentiment"]], use_container_width=True)
                
                col1, col2 = st.columns([0.6, 0.4])
                with col1:
                    st.subheader("📊 Sentiment Distribution")
                    chart_data = predictions_df['sentiment'].value_counts().reset_index()
                    chart_data.columns = ['sentiment', 'count']
                    chart = alt.Chart(chart_data).mark_bar().encode(
                        x=alt.X('sentiment', axis=alt.Axis(title='Sentiment')),
                        y=alt.Y('count', axis=alt.Axis(title='Articles')),
                        color='sentiment',
                        tooltip=['sentiment', 'count']
                    ).properties(title='Predicted News Sentiment')
                    st.altair_chart(chart, use_container_width=True)
                
                with col2:
                    st.subheader("📡 Live Stream Simulation")
                    placeholder = st.empty()
                    for idx, row in predictions_df.iterrows():
                        sentiment_emoji = {"Positive": "😄", "Neutral": "😐", "Negative": "😠"}
                        with placeholder.container():
                            st.markdown(f"**{row['title']}**")
                            st.markdown(f"{row['sentiment']} {sentiment_emoji.get(row['sentiment'], '')}")
                            st.progress((idx + 1) / len(predictions_df))
                        time.sleep(0.5)
                        
                st.balloons()
else:
    st.info("👈 Adjust controls and click 'Run Analysis'")
