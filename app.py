from pathlib import Path
import re

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).parent
NEED_RULES = {
    "Discounts and offers": ["discount", "sale", "offer", "coupon", "voucher", "promo", "kam price", "price kam", "sasta", "سست", "رعایت", "ڈسکاؤنٹ", "آفر", "سیل"],
    "Delivery and order status": ["delivery", "deliver", "shipping", "courier", "dispatch", "parcel", "order status", "late", "delay", "waiting", "kitne din", "kitnay din", "din mein", "بھیج", "ڈیلیوری", "ترسیل", "کورئیر", "پہنچ"],
    "Sizing and fit": ["size", "sizing", "small", "medium", "large", "xl", "2xl", "fit", "measure", "waist", "chest", "سائز", "ناپ", "میڈیم", "چھوٹا", "بڑا"],
    "Fabric and product quality": ["fabric", "quality", "material", "cloth", "kapra", "kapray", "stitch", "silai", "embroidery", "design", "colour", "color", "cotton", "lawn", "print", "کپڑا", "معیار", "کوالٹی", "سلائی", "کپڑے"],
    "Availability and location": ["available", "availability", "stock", "restock", "out of stock", "in stock", "karachi", "lahore", "islamabad", "location", "store", "shop", "kahan se", "milay ga", "milega", "دستیاب", "کہاں", "اسٹاک", "کراچی", "لاہور", "اسلام آباد"],
    "Purchase and ordering": ["order", "buy", "purchase", "inbox", "dm", "send link", "order kar", "book kar", "details send", "لنک", "آرڈر", "خرید", "ان باکس", "ڈی ایم"],
    "Price inquiry": ["price", "cost", "rate", "qeemat", "keemat", "daam", "kitne ki", "kitnay ki", "kitna ka", "rupees", "pkr", "مہنگا", "قیمت", "دام", "کتنے کی", "کتنے کا", "روپے"],
    "Positive feedback": ["beautiful", "lovely", "gorgeous", "amazing", "awesome", "nice dress", "love this", "masha allah", "mashallah", "zabardast", "khubsurat", "pyara", "خوبصورت", "ماشاء اللہ", "زبردست", "اچھا"],
}
ROMAN_URDU_MARKERS = ["bohat", "bohot", "hai", "hain", "kya", "kia", "kapra", "kapray", "milega", "milay", "kitne", "kitnay", "kar dein", "bata dein", "acha", "achi", "keemat", "qeemat"]

st.set_page_config(page_title="Pakistan Social Audience Intelligence", page_icon="📊", layout="wide")
st.title("Pakistan Social Audience & Campaign Intelligence")
st.caption("Compare available video formats, discover comment needs, and turn findings into marketing tests.")


def has_term(text, term):
    if re.search(r"[A-Za-z0-9]", term):
        return re.search(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])", text, flags=re.IGNORECASE) is not None
    return term in text


def infer_need(text):
    normalized = str(text).casefold()
    for need, terms in NEED_RULES.items():
        found = [term for term in terms if has_term(normalized, term)]
        if found:
            return need, ", ".join(found[:3]), "keyword match"
    return "Other / needs review", "", "unmatched"


def infer_language(text):
    value = str(text)
    if re.search(r"[\u0600-\u06ff]", value):
        return "Urdu script (detected)"
    lowered = value.casefold()
    if any(has_term(lowered, token) for token in ROMAN_URDU_MARKERS):
        return "Roman Urdu / English (heuristic)"
    return "Latin script; language uncertain"


@st.cache_data
def supplied_data():
    """Load P1 observed YouTube exports and separate P2 synthetic planning summaries."""
    raw_posts = pd.read_csv(ROOT / "data/p1_youtube_video_analytics.csv")
    raw_comments = pd.read_csv(ROOT / "data/p1_youtube_comments_sentiment.csv")
    raw_platforms = pd.read_csv(ROOT / "data/p2_platform_performance_synthetic.csv")
    raw_campaigns = pd.read_csv(ROOT / "data/p2_campaign_performance_synthetic.csv")
    raw_budget_plan = pd.read_csv(ROOT / "data/p2_budget_plan_synthetic.csv")
    dates = pd.to_datetime(raw_posts["published_at"], errors="coerce", utc=True)
    titles = raw_posts["title"].fillna("").astype(str)
    posts = pd.DataFrame({
        "post_id": raw_posts["video_id"],
        "brand": raw_posts["channel"],
        "platform": "YouTube",
        "campaign": "Public YouTube fashion sample",
        "content_type": titles.str.contains(r"#?shorts|#short", case=False).map({True: "Short-form video (Shorts)", False: "Standard video"}),
        "product_category": "Pakistani fashion video",
        "likes": raw_posts["likes"],
        "comments_count": raw_posts["comment_count"],
        "views": raw_posts["views"],
        "days_since_post": (pd.Timestamp.now(tz="UTC") - dates).dt.days.clip(lower=0).fillna(0),
        "source_url": raw_posts["url"] if "url" in raw_posts else "",
        "data_origin": "P1 supplied YouTube data",
    })
    comments = pd.DataFrame({
        "comment_id": [f"P1C{i:05d}" for i in range(1, len(raw_comments) + 1)],
        "post_id": raw_comments["video_id"],
        "comment_text": raw_comments["comment"].fillna("").astype(str),
        "sentiment_p1": raw_comments.get("sentiment_ml", "Not provided"),
        "published_at": raw_comments.get("published_at", ""),
        "topic": "",
        "data_origin": "P1 supplied YouTube data",
    })
    # These P2 aggregates are synthetic classroom data. Keep them separate from P1 records.
    return posts, comments, raw_platforms, raw_campaigns, raw_budget_plan


def custom_data(post_file, comment_file):
    posts = pd.read_csv(post_file)
    comments = pd.read_csv(comment_file)
    required_posts = ["post_id", "brand", "platform", "campaign", "content_type", "product_category", "likes", "comments_count"]
    required_comments = ["post_id", "comment_text"]
    missing_posts = [column for column in required_posts if column not in posts.columns]
    missing_comments = [column for column in required_comments if column not in comments.columns]
    if missing_posts or missing_comments:
        raise ValueError("Missing post columns: " + ", ".join(missing_posts or ["none"]) + ". Missing comment columns: " + ", ".join(missing_comments or ["none"]) + ". See README for schemas.")
    for column, default in {"views": 0, "shares": 0, "days_since_post": 0, "source_url": ""}.items():
        if column not in posts:
            posts[column] = default
    for column, default in {"comment_id": None, "sentiment_ml": "Not supplied", "published_at": "", "topic": ""}.items():
        if column not in comments:
            comments[column] = default
    if "language_type" not in comments:
        comments["language_type"] = comments.comment_text.map(infer_language)
    posts["data_origin"] = "P3 user-supplied CSV"
    comments["data_origin"] = "P3 user-supplied CSV"
    return posts, comments


@st.cache_data(show_spinner=False)
def cluster_unmatched(texts, cluster_count):
    if len(texts) < 2:
        return [0] * len(texts), {0: "Unmatched comments"}
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=1800, min_df=1, strip_accents="unicode")
    matrix = vectorizer.fit_transform(texts)
    if matrix.shape[1] == 0 or matrix.nnz == 0:
        return [0] * len(texts), {0: "Short or unrecognized comments"}
    count = min(cluster_count, len(texts))
    labels = KMeans(n_clusters=count, random_state=42, n_init=5).fit_predict(matrix)
    terms = vectorizer.get_feature_names_out()
    names = {}
    for group in set(labels):
        mean = matrix[labels == group].mean(axis=0).A1
        top = mean.argsort()[::-1][:4]
        keywords = [terms[i].strip() for i in top if mean[i] > 0 and len(terms[i].strip()) >= 3]
        names[int(group)] = "Unmatched cluster: " + (", ".join(keywords[:3]) if keywords else f"group {group + 1}")
    return labels.tolist(), names


def assign_audience_needs(comments, cluster_count):
    output = comments.copy()
    inferred = output["comment_text"].map(infer_need)
    output[["audience_need", "matched_terms", "classification_method"]] = pd.DataFrame(inferred.tolist(), index=output.index)
    detected_language = output["comment_text"].map(infer_language)
    if "language_type" in output:
        existing_language = output["language_type"].fillna("").astype(str).str.strip()
        output["language_type"] = existing_language.where(existing_language.ne(""), detected_language)
    else:
        output["language_type"] = detected_language
    remaining = output["audience_need"].eq("Other / needs review")
    if remaining.any():
        fallback_labels, fallback_names = cluster_unmatched(output.loc[remaining, "comment_text"].tolist(), cluster_count)
        idx = output.index[remaining]
        output.loc[idx, "cluster_id"] = fallback_labels
        output.loc[idx, "cluster_name"] = [fallback_names[label] for label in fallback_labels]
        output.loc[idx, "audience_need"] = output.loc[idx, "cluster_name"]
        output.loc[idx, "classification_method"] = "unsupervised clue; review manually"
    output["cluster_id"] = output.get("cluster_id", pd.Series(index=output.index, dtype="object"))
    return output


posts, comments, p2_platforms, p2_campaigns, p2_budget_plan = supplied_data()
with st.sidebar:
    st.header("Data")
    data_source = st.radio("Choose post and comment data", ["P1 supplied YouTube data", "CSV files in data folder", "Upload CSV files"])
    st.caption("P1 is YouTube-only. P2 campaign and platform figures are synthetic and stay separate.")
    custom_posts = None
    custom_comments = None
    if data_source == "CSV files in data folder":
        post_path = ROOT / "data/custom_posts.csv"
        comment_path = ROOT / "data/custom_comments.csv"
        if post_path.exists() and comment_path.exists():
            custom_posts, custom_comments = post_path, comment_path
            st.success("Found both custom CSV files in data folder.")
        else:
            st.warning("Place custom_posts.csv and custom_comments.csv in the data folder first.")
    elif data_source == "Upload CSV files":
        custom_posts = st.file_uploader("Posts CSV", type="csv", key="custom_posts")
        custom_comments = st.file_uploader("Comments CSV", type="csv", key="custom_comments")

if data_source != "P1 supplied YouTube data":
    if custom_posts is None or custom_comments is None:
        st.info("Add both CSV files or select P1 supplied YouTube data to continue.")
        st.stop()
    try:
        posts, comments = custom_data(custom_posts, custom_comments)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

if posts.empty:
    st.error("The selected posts file contains no records.")
    st.stop()
platform_choices = sorted(posts.platform.fillna("Unknown").astype(str).unique().tolist())
with st.sidebar:
    selected_platform = st.selectbox("Platform for comparison", platform_choices)
    platform_posts = posts[posts.platform.fillna("Unknown").astype(str).eq(selected_platform)].copy()
    campaign_choices = sorted(platform_posts.campaign.fillna("Unknown").astype(str).unique().tolist())
    selected_campaign = st.selectbox("Campaign", ["All campaigns"] + campaign_choices)
    if selected_campaign != "All campaigns":
        platform_posts = platform_posts[platform_posts.campaign.fillna("Unknown").astype(str).eq(selected_campaign)]
posts = platform_posts
comments = comments[comments.post_id.isin(posts.post_id)].copy()
comments = comments[comments.comment_text.fillna("").astype(str).str.strip().ne("")].copy()
if comments.empty:
    st.warning("No comments are linked to the selected platform and campaign. Choose a different filter or check that post IDs match between the CSVs.")
    st.stop()
comments = assign_audience_needs(comments, 6)
provided_topics = comments.topic.fillna("").astype(str).str.strip()
has_provided_topic = provided_topics.ne("") & ~provided_topics.str.casefold().isin(["not labeled in source", "nan", "none"])
comments.loc[has_provided_topic, "audience_need"] = provided_topics[has_provided_topic]
comments.loc[has_provided_topic, "classification_method"] = "provided topic label"
for col in ["likes", "comments_count", "views", "days_since_post"]:
    posts[col] = pd.to_numeric(posts[col], errors="coerce").fillna(0)
posts["shares"] = pd.to_numeric(posts.get("shares", 0), errors="coerce")
posts["engagement_rate"] = ((posts.likes + posts.comments_count + posts.shares.fillna(0)) / posts.views.where(posts.views > 0)) * 100

# Compare outliers within platform and format. Broaden the peer group only when a slice is too small.
posts["anomaly_score"] = 0.0
for (platform, content_type), group in posts.groupby(["platform", "content_type"]):
    peer = group if len(group) >= 8 else posts[posts.platform.eq(platform)]
    if len(peer) < 8:
        peer = posts[posts.content_type.eq(content_type)]
    if len(peer) < 8:
        peer = posts
    center = peer.engagement_rate.median()
    mad = (peer.engagement_rate - center).abs().median()
    posts.loc[group.index, "anomaly_score"] = (group.engagement_rate - center).abs() / max(1.4826 * mad, 0.01)
posts["investigate"] = posts.anomaly_score >= 3.5

audience_summary = comments.groupby("audience_need", as_index=False).agg(comments=("comment_text", "size"), method=("classification_method", "first")).sort_values("comments", ascending=False)
content_summary = posts.groupby("content_type", as_index=False).agg(
    posts=("post_id", "count"),
    median_engagement_rate=("engagement_rate", "median"),
    median_views=("views", "median"),
    median_comments=("comments_count", "median"),
).sort_values("median_engagement_rate", ascending=False)
using_p1 = posts.data_origin.eq("P1 supplied YouTube data").all()
source_label = "P1 YouTube" if using_p1 else "Selected dataset"

m1, m2, m3, m4 = st.columns(4)
m1.metric(f"{source_label} posts", f"{len(posts):,}")
m2.metric(f"{source_label} comments", f"{len(comments):,}")
m3.metric("Audience need groups", f"{len(audience_summary):,}")
m4.metric("Unusual posts to review", int(posts.investigate.sum()))
if using_p1:
    st.info("Selected data: P1 YouTube records. P2 channel and campaign files are synthetic planning examples and are not combined with P1 observations.")
else:
    st.info(f"Selected custom data: {posts.platform.nunique()} platform(s), {posts.content_type.nunique()} content format(s). P2 channel and campaign files are synthetic planning examples and remain separate.")

tabs = st.tabs(["Compare content types", "Discover audience needs", "Unusual patterns", "Recommendations", "P2 marketing context"])
with tabs[0]:
    st.subheader("Compare available content formats")
    st.plotly_chart(px.bar(content_summary, x="content_type", y="median_engagement_rate", color="content_type", text="posts", labels={"content_type": "Content format", "median_engagement_rate": "Median engagement rate (%)", "posts": "Posts"}), use_container_width=True)
    st.dataframe(content_summary, use_container_width=True, hide_index=True)
    if using_p1:
        st.caption(f"This comparison uses P1 YouTube videos from the {selected_campaign} selection. Shorts are inferred from title/hashtag text; all other records are grouped as standard videos. P1 does not include Instagram Reels, photos, influencer labels, Eid/campaign labels, or shares. Rate = (likes + comments) / views.")
    else:
        st.caption(f"Content types come from the selected CSVs. Current comparison: {selected_platform} / {selected_campaign}. Rate = (likes + comments + available shares) / views. Compare like formats within comparable platforms and post ages; missing views produce no engagement rate.")

with tabs[1]:
    st.subheader("Discover what commenters ask about")
    st.write("The app first applies transparent English, Roman Urdu, and Urdu-script keyword rules. Comments without a rule match are grouped by character n-gram TF-IDF and K-Means to suggest themes for manual review.")
    st.plotly_chart(px.bar(audience_summary.head(15).sort_values("comments"), x="comments", y="audience_need", orientation="h", color="method", labels={"comments": "Comments", "audience_need": "Audience need group", "method": "How group was formed"}), use_container_width=True)
    left, right = st.columns([1, 2])
    with left:
        st.markdown("**Detected script / language clue**")
        st.dataframe(comments.language_type.value_counts().rename_axis("inferred label").reset_index(name="comments"), use_container_width=True, hide_index=True)
    with right:
        selected_need = st.selectbox("Inspect a need group", audience_summary.audience_need.tolist())
        examples = comments.loc[comments.audience_need.eq(selected_need), ["comment_text", "matched_terms", "language_type", "classification_method", "post_id"]].head(30)
        st.dataframe(examples, use_container_width=True, hide_index=True)
    st.caption("The P1 source has no human topic or language labels. Language/script labels and rule matches are inferred; unsupervised clusters are clues, not validated customer segments. Review sample comments before presenting these counts as research findings.")

with tabs[2]:
    st.subheader("Posts with unusual engagement for their format")
    flagged = posts[posts.investigate].sort_values("anomaly_score", ascending=False)
    st.write(f"{len(flagged)} videos crossed the robust outlier threshold within their content format.")
    st.dataframe(flagged[["post_id", "brand", "content_type", "engagement_rate", "anomaly_score", "views", "comments_count", "source_url"]], use_container_width=True, hide_index=True)
    st.caption("Flags are review prompts, not evidence of manipulation or success. The source is YouTube-only and does not provide campaign, paid-promotion, or full comment-topic history per video.")

with tabs[3]:
    st.subheader("Evidence-based next actions")
    best_format = content_summary.iloc[0]
    leading_need = audience_summary.iloc[0]
    need_actions = {
        "Discounts and offers": "Test a clearly stated PKR offer against a no-discount message, and compare qualified inquiries or conversions.",
        "Delivery and order status": "Publish delivery timelines, city coverage, and order-status instructions; track repeated delivery questions.",
        "Sizing and fit": "Add a size chart and fit demonstration to product posts, then monitor sizing questions.",
        "Fabric and product quality": "Show close-up fabric, stitching, and care details; test which proof points reduce quality questions.",
        "Availability and location": "Show current colors, sizes, stock, store locations, and restock timing in the caption or pinned comment.",
        "Purchase and ordering": "Make the order path, official link, and response channel explicit; measure completed orders rather than likes.",
        "Price inquiry": "Show the PKR price and what's included earlier in the caption or video.",
        "Positive feedback": "Use these comments as candidate social-proof material only after reviewing permission and platform rules.",
    }
    st.markdown(f"- **Format test from {source_label}:** {best_format.content_type} on {selected_platform} has the highest median observed engagement rate ({best_format.median_engagement_rate:.2f}%) among the selected formats. Run a comparable follow-up test before scaling.")
    st.markdown(f"- **Audience need from P1:** {leading_need.audience_need} is the largest detected group ({int(leading_need.comments):,} comments). {need_actions.get(leading_need.audience_need, 'Review representative comments in this group and manually name it before selecting a response.')}")
    top_p2 = p2_platforms.sort_values("ROAS", ascending=False).iloc[0]
    st.markdown(f"- **Channel test from synthetic P2:** {top_p2.platform} has the highest simulated ROAS ({top_p2.ROAS:.2f}x) in the P2 planning data. Treat this only as a hypothesis to test on real campaign results.")
    if len(flagged):
        st.markdown(f"- **Quality review:** inspect the {len(flagged)} flagged video(s), checking format, age, channel size, and possible promotion before interpreting them.")
    st.warning("P2 outcomes are simulated and do not establish real channel performance. P1 and P2 describe different samples and are intentionally not joined.")

with tabs[4]:
    st.subheader("P2 campaign and budget context — synthetic classroom data")
    st.warning("P2’s README identifies the advertising records as synthetic. These figures are planning examples, not observed brand results.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Simulated platform outcomes**")
        st.plotly_chart(px.bar(p2_platforms.sort_values("ROAS", ascending=False), x="platform", y="ROAS", color="platform", labels={"ROAS": "Simulated ROAS (x)"}), use_container_width=True)
        st.dataframe(p2_platforms[["platform", "spend", "conversions", "revenue", "ROAS", "CVR"]].sort_values("ROAS", ascending=False), use_container_width=True, hide_index=True)
    with col2:
        st.markdown("**Simulated campaign outcomes**")
        st.plotly_chart(px.bar(p2_campaigns.sort_values("ROAS", ascending=False), x="campaign", y="ROAS", color="campaign", labels={"ROAS": "Simulated ROAS (x)"}), use_container_width=True)
        st.dataframe(p2_campaigns[["campaign", "spend", "conversions", "revenue", "ROAS", "CVR"]].sort_values("ROAS", ascending=False), use_container_width=True, hide_index=True)
    st.markdown("**Example simulated budget plan**")
    st.dataframe(p2_budget_plan[["Platform", "ROAS (x)", "Conversion rate (%)", "Allocation (%)", "Recommended budget (PKR)"]].sort_values("Allocation (%)", ascending=False), use_container_width=True, hide_index=True)

st.divider()
st.download_button("Download comments with inferred needs", comments.to_csv(index=False).encode("utf-8-sig"), "p1_audience_needs.csv", "text/csv")
st.download_button("Download P1 video comparison and outlier scores", posts.to_csv(index=False).encode("utf-8-sig"), "p1_video_analysis.csv", "text/csv")
st.caption("Project 3 • P1 = supplied YouTube records • P2 = separate synthetic advertising-planning data")
