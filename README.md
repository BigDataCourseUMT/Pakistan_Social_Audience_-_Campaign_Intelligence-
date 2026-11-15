# Pakistan Social Audience and Campaign Intelligence

A Streamlit project that uses the supplied P1 Pakistani fashion YouTube data to compare video formats, group similar comment needs, flag unusual engagement patterns, and turn findings into practical test ideas. The clustering is unsupervised; this project complements rather than replaces Project 1 sentiment classification.

## What's in this folder

- `app.py` — interactive dashboard and analysis
- `data/p1_youtube_video_analytics.csv` — P1 video analytics export (525 videos)
- `data/p1_youtube_comments_sentiment.csv` — P1 comments export (8,641 comments)
- `requirements.txt` — Python dependencies

**Data scope:** the default data is copied from the project's P1 YouTube exports. It is not Instagram or TikTok data. Shares, language labels, and purchase-intent labels are not present in the supplied CSVs. The app leaves these unavailable or unlabeled and does not treat them as observed values. The P1 sentiment prediction column is retained as a reference, but clustering is based on comment text.

## 1. Install VS Code and Python

1. Install [Visual Studio Code](https://code.visualstudio.com/) and Python 3.12. In VS Code, install the Microsoft **Python** extension. Check that `python --version` reports Python 3.12.
2. Open VS Code, choose **File → Open Folder**, and select this `Pakistani_Social_Audience_Campaign_Intelligence` folder inside P3.
3. Open **Terminal → New Terminal**. In PowerShell, create and activate a project environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

   If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once in a normal user PowerShell, then activate again.
4. In VS Code, press `Ctrl+Shift+P`, choose **Python: Select Interpreter**, and select `.venv`.

## 2. Run the app locally

```powershell
streamlit run app.py
```

The app opens in a browser at `http://localhost:8501`. Keep the terminal open while using it. The default source is the supplied P1 YouTube data. Stop the server with `Ctrl+C` in the terminal.

## 3. Prepare permitted data

You can upload two CSV files from the sidebar. Required columns:

**Posts CSV:** `post_id, brand, platform, campaign, content_type, product_category, likes, comments_count, shares, views, days_since_post`

**Comments CSV:** `comment_id, post_id, comment_text, language_type, topic, sentiment, purchase_intent`

`topic`, `sentiment`, and `purchase_intent` can be included for your research records; the clustering uses comment text, not those labels. Numeric engagement columns should contain numbers. Use a consistent post ID between the two files. Missing/unavailable metrics may be left blank. Remove names, handles, phone numbers, and other personal data before upload. Uploads are processed in the current app session and are not saved by this app.

## 4. GitHub repository

The project is published at [BigDataCourseUMT/Pakistan_Social_Audience_-_Campaign_Intelligence-](https://github.com/BigDataCourseUMT/Pakistan_Social_Audience_-_Campaign_Intelligence-). The local `main` branch tracks that repository. Do not commit `.venv`, private customer data, or access tokens. `.gitignore` excludes the local environment and secrets files.

For future updates:

```powershell
git add .
git commit -m "Describe your change"
git push
```

## 5. Deploy on Streamlit Community Cloud

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub and grant access to the repository. Deployment requires admin access to that GitHub repository.
2. Click **Create app**, select the repository, branch `main`, and main file path `app.py`.
3. Select a supported Python version (3.11 or 3.12), then deploy. Community Cloud installs dependencies from `requirements.txt`.
4. Open the generated `*.streamlit.app` URL and check the P1 data source and dashboard tabs. Each push to the deployed branch updates the app.

No API key or secrets are needed for this version. If you add credentials later, store them in Streamlit's **Advanced settings → Secrets**, and never commit them to GitHub.

## How to explain the analysis

- **Content performance:** median engagement rate by content format. The P1 YouTube source has no share metric, so rate = `(likes + comments) / views × 100`; this comparison is descriptive and may be biased by video age, channel size, paid promotion, and missing metrics.
- **Audience groups:** TF-IDF vectorization and K-Means group similar comment text. The displayed words are clues for manual interpretation; clusters are not validated customer segments, and Roman Urdu/Urdu results need human review.
- **Unusual posts:** a robust median-absolute-deviation threshold flags engagement rates that differ from the dataset. A flag is a review prompt, not evidence of manipulation or success.
- **Recommendations:** suggest candidate content and message tests from observed patterns. Use controlled comparisons and sales or qualified inquiry outcomes to assess results.

## Project scope and next steps

This is a classroom-ready prototype rather than a production analytics system. The P1 export provides 525 YouTube videos and 8,641 comments; comments are available for 327 unique video IDs. The brief discusses cross-platform public social posts, while the supplied P1 files cover YouTube only. For the final report, state this scope clearly, manually review a sample of clusters, compare Shorts with Shorts and standard videos with standard videos, and describe missing data and limitations.

## Windows App Control error

If importing pandas fails with `DLL load failed while importing _compute` and says an Application Control policy blocked the file, Windows is blocking PyArrow's native library. Reinstalling packages in the same environment does not remove that policy. Ask your device administrator or university IT to approve the blocked dependency. The app can still run in Streamlit Community Cloud, which uses its own hosted environment.
