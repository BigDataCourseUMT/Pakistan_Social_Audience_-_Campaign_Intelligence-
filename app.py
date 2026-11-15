from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).parent
POST_COLUMNS = ["post_id", "brand", "platform", "campaign", "content_type", "product_category", "likes", "comments_count", "shares", "views", "days_since_post", "data_origin"]
COMMENT_COLUMNS = ["comment_id", "post_id", "comment_text", "language_type", "topic", "sentiment", "purchase_intent", "data_origin"]
st.set_page_config(page_title="Pakistan Social Audience Intelligence", page_icon="📊", layout="wide")
st.title("Pakistan Social Audience & Campaign Intelligence")
st.caption("Discover audience needs, content patterns, and posts to investigate for Pakistani fashion campaigns.")

@st.cache_data
def supplied_data():
    """Adapt the P1 YouTube exports to the common analysis schema."""
    raw_posts = pd.read_csv(ROOT / "data/p1_youtube_video_analytics.csv")
    raw_comments = pd.read_csv(ROOT / "data/p1_youtube_comments_sentiment.csv")
    dates = pd.to_datetime(raw_posts["published_at"], errors="coerce", utc=True)
    posts = pd.DataFrame({
        "post_id": raw_posts["video_id"],
        "brand": raw_posts["channel"],
        "platform": "YouTube",
        "campaign": "Public YouTube fashion sample",
        "content_type": raw_posts["title"].fillna("").str.contains(r"#?shorts|#short", case=False).map({True: "Shorts", False: "Standard video"}),
        "product_category": "Pakistani fashion video",
        "likes": raw_posts["likes"],
        "comments_count": raw_posts["comment_count"],
        # P1 source has no share metric; 0 is used only as an unavailable placeholder.
        "shares": 0,
        "views": raw_posts["views"],
        "days_since_post": (pd.Timestamp.now(tz="UTC") - dates).dt.days.clip(lower=0).fillna(0),
        "data_origin": "P1 supplied YouTube data",
        "url": raw_posts["url"] if "url" in raw_posts else "",
    })
    comments = pd.DataFrame({
        "comment_id": [f"P1C{i:05d}" for i in range(1, len(raw_comments) + 1)],
        "post_id": raw_comments["video_id"],
        "comment_text": raw_comments["comment"],
        "language_type": "Not provided in source",
        "topic": "Not labeled in source",
        "sentiment": raw_comments.get("sentiment_ml", "Not provided"),
        "purchase_intent": "Not labeled in source",
        "data_origin": "P1 supplied YouTube data",
        "published_at": raw_comments.get("published_at", ""),
    })
    return posts, comments

def clean_frame(df, required, label):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} file is missing required columns: {', '.join(missing)}")
    return df

@st.cache_data(show_spinner=False)
def cluster_comments(texts, k):
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=1200, min_df=1, strip_accents="unicode")
    matrix = vec.fit_transform(texts)
    if matrix.shape[0] < 2 or matrix.nnz == 0:
        return [0] * len(texts), {0: "General / short comments"}
    k = min(k, matrix.shape[0])
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(matrix)
    terms = vec.get_feature_names_out()
    names = {}
    for group in set(labels):
        mean = matrix[labels == group].mean(axis=0).A1
        top = mean.argsort()[::-1][:3]
        names[int(group)] = ", ".join(terms[i] for i in top if mean[i] > 0) or f"Group {group + 1}"
    return labels.tolist(), names

with st.sidebar:
    st.header("Data")
    source = st.radio("Choose input", ["P1 supplied YouTube data", "Upload my CSV files"])
    st.caption("Default data comes from P1. The source includes YouTube videos and comments, not Instagram or TikTok records.")
    k = st.slider("Comment clusters", 2, 10, 6)

try:
    if source == "Upload my CSV files":
        pfile = st.file_uploader("Posts CSV", type="csv", key="posts")
        cfile = st.file_uploader("Comments CSV", type="csv", key="comments")
        if not pfile or not cfile:
            st.info("Upload both files. Download the column templates from the README or use the sample CSVs as a guide.")
            st.stop()
        posts = clean_frame(pd.read_csv(pfile), POST_COLUMNS[:-1], "Posts")
        comments = clean_frame(pd.read_csv(cfile), COMMENT_COLUMNS[:-1], "Comments")
        posts["data_origin"] = "user_uploaded"
        comments["data_origin"] = "user_uploaded"
    else:
        posts, comments = supplied_data()
except Exception as exc:
    st.error(str(exc))
    st.stop()

for col in ["likes", "comments_count", "shares", "views", "days_since_post"]:
    posts[col] = pd.to_numeric(posts[col], errors="coerce").fillna(0)
comments["comment_text"] = comments["comment_text"].fillna("").astype(str)
comments = comments[comments.comment_text.str.strip().ne("")].copy()
if comments.empty or posts.empty:
    st.warning("The uploaded files need at least one post and one non-empty comment.")
    st.stop()

labels, names = cluster_comments(comments.comment_text.tolist(), k)
comments["cluster_id"] = labels
comments["cluster_name"] = comments.cluster_id.map(names)
posts["engagement_rate"] = (posts.likes + posts.comments_count + posts.shares) / posts.views.clip(lower=1) * 100
comments_per_post = comments.groupby("post_id").size().rename("observed_comments")
posts = posts.merge(comments_per_post, on="post_id", how="left")
posts["observed_comments"] = posts.observed_comments.fillna(0)
median = posts.engagement_rate.median()
mad = (posts.engagement_rate - median).abs().median()
posts["anomaly_score"] = (posts.engagement_rate - median).abs() / max(1.4826 * mad, 0.01)
posts["investigate"] = posts.anomaly_score >= 3.5

total1, total2, total3, total4 = st.columns(4)
total1.metric("Posts", f"{len(posts):,}")
total2.metric("Comments analyzed", f"{len(comments):,}")
total3.metric("Audience groups", comments.cluster_id.nunique())
total4.metric("Posts to investigate", int(posts.investigate.sum()))
if posts.data_origin.eq("P1 supplied YouTube data").any():
    st.info(f"Using P1 source data: {len(posts):,} YouTube videos and {len(comments):,} comments. Shares, language labels, and purchase-intent labels are not present in the provided source.")

tab1, tab2, tab3, tab4 = st.tabs(["Content performance", "Audience needs", "Unusual posts", "Recommendations"])
with tab1:
    st.subheader("Which content attracts attention?")
    perf = posts.groupby("content_type", as_index=False).agg(posts=("post_id", "count"), median_engagement_rate=("engagement_rate", "median"), avg_views=("views", "mean"), avg_comments=("comments_count", "mean"))
    st.plotly_chart(px.bar(perf, x="content_type", y="median_engagement_rate", color="content_type", labels={"median_engagement_rate": "Median engagement rate (%)", "content_type": "Content format"}), use_container_width=True)
    st.dataframe(perf.sort_values("median_engagement_rate", ascending=False), use_container_width=True, hide_index=True)
    st.caption("P1 does not provide shares, so this rate is (likes + comments) / views. Compare Shorts with Shorts and standard videos with standard videos; views and comments can depend on video age and channel size.")
with tab2:
    st.subheader("What does the audience want?")
    cl = comments.groupby(["cluster_id", "cluster_name"], as_index=False).size().sort_values("size", ascending=False)
    a, b = st.columns([1, 1])
    with a: st.plotly_chart(px.bar(cl, x="size", y="cluster_name", orientation="h", labels={"size": "Comments", "cluster_name": "Top terms in cluster"}), use_container_width=True)
    with b: st.dataframe(comments[["comment_text", "cluster_name", "language_type", "post_id"]].head(100), use_container_width=True, hide_index=True)
    st.caption("Clusters are unsupervised TF-IDF and K-Means groups. P1 does not include language labels. Read representative comments and manually validate groups before reporting them as customer segments.")
with tab3:
    st.subheader("Which marketing patterns are unusual?")
    flagged = posts[posts.investigate].sort_values("anomaly_score", ascending=False)
    st.write(f"Flagged {len(flagged)} posts using a robust median absolute deviation screen on engagement rate.")
    st.dataframe(flagged[["post_id", "brand", "platform", "campaign", "content_type", "engagement_rate", "anomaly_score", "observed_comments"]], use_container_width=True, hide_index=True)
    st.caption("A flag is a prompt for human review, not proof of fraud or campaign success. Compare like with like and check data quality.")
with tab4:
    st.subheader("What should the team do next?")
    winners = perf.sort_values("median_engagement_rate", ascending=False)
    biggest = cl.iloc[0] if len(cl) else None
    if len(winners): st.markdown(f"- **Content test:** use **{winners.iloc[0].content_type}** as a candidate format, then run a controlled test against a comparable alternative.")
    if biggest is not None: st.markdown(f"- **Audience response:** inspect the largest comment group (top terms: **{biggest.cluster_name}**); draft a relevant FAQ or product detail and validate it on sampled comments.")
    needs = comments.groupby("cluster_name").size().sort_values(ascending=False)
    if len(needs) > 1: st.markdown(f"- **Campaign experiment:** test one offer or message addressing **{needs.index[1]}** and compare conversion or qualified inquiries, not likes alone.")
    if len(flagged): st.markdown(f"- **Quality review:** manually inspect the {len(flagged)} flagged post(s), their age, platform, paid promotion, and comment mix before drawing conclusions.")
    st.info("Recommendations are descriptive and exploratory. This dashboard does not measure causality, sales conversion, or verified individual customer segments.")

st.divider()
st.download_button("Download analyzed comments", comments.to_csv(index=False).encode("utf-8-sig"), "audience_clusters.csv", "text/csv")
st.download_button("Download post analysis", posts.to_csv(index=False).encode("utf-8-sig"), "post_performance.csv", "text/csv")
st.caption("Project 3 • Pakistan social audience and campaign intelligence • default source: P1 YouTube exports")
