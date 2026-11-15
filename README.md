# Pakistan Social Audience and Campaign Intelligence

A Streamlit project for comparing available content formats, finding audience needs in comments, flagging unusual engagement, and turning those findings into marketing tests. It uses P1's supplied YouTube records by default and keeps P2's simulated campaign-planning summaries separate.

## What's in this folder

- `app.py` — interactive dashboard and analysis
- `data/p1_youtube_video_analytics.csv` — P1 video analytics export (525 videos)
- `data/p1_youtube_comments_sentiment.csv` — P1 comments export (8,641 comments)
- `data/p2_platform_performance_synthetic.csv` — P2 simulated platform advertising results
- `data/p2_campaign_performance_synthetic.csv` — P2 simulated campaign results
- `data/p2_budget_plan_synthetic.csv` — P2 simulated allocation plan
- `data/p2_platform_performance_synthetic.csv` — P2 simulated platform advertising results
- `data/p2_campaign_performance_synthetic.csv` — P2 simulated campaign results
- `data/p2_budget_plan_synthetic.csv` — P2 simulated budget allocation
- `requirements.txt` — Python dependencies

**Data scope:** P1 provides YouTube records, not Instagram, TikTok, or Facebook posts. P2's README describes its advertising data as synthetic; its platform and campaign summaries are shown separately and never merged with P1 observations. The default format comparison can distinguish YouTube Shorts from standard videos only. Add permitted real posts/comments as described below for Reels, product photos, influencer content, sale/Eid campaigns, and cross-platform needs analysis. P1 lacks shares and hand-labeled comment topics/languages, so the app marks them unavailable or inferred.

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

## 3. Compare more content types and discover audience needs

The default view compares Shorts with standard YouTube videos. To compare the brief's formats across Instagram, TikTok, or Facebook, use permitted records from those platforms. Put them in `data/custom_posts.csv` and `data/custom_comments.csv`, then select **CSV files in data folder**; or choose **Upload CSV files** in the sidebar.

Posts must include `post_id, brand, platform, campaign, content_type, product_category, likes, comments_count`. Add `views`, `shares`, `days_since_post`, and `source_url` where available.

Comments must include `post_id, comment_text`. Add `comment_id`, `topic`, `language_type`, `sentiment_ml`, and `published_at` when available. Keep the same post ID in both CSVs. If a real, human-reviewed `topic` is supplied, the app uses it; otherwise it applies transparent English/Roman Urdu/Urdu-script keyword rules for price, discount, delivery, sizing, product quality, availability, purchase intent, and positive feedback. It groups unmatched comments with character n-gram TF-IDF and K-Means to suggest themes. Review the examples and counts before reporting inferred needs as findings.

For a meaningful cross-platform content-format comparison, include real posts from each format (for example Reels, product photos, influencer posts, styling videos, and sale/Eid posts), plus campaign, platform, and post age fields. If views are unavailable, engagement-rate comparisons will be unavailable for those rows. Record source URL, collection date, and reuse permission in your research notes. Remove private customer identifiers before upload; uploaded files are processed for the current session and are not saved by the app.

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
4. Open the generated `*.streamlit.app` URL and check the content comparison, audience-need groups, and P2 synthetic-planning tab. Each push to the deployed branch updates the app.

No API key or secrets are needed for this version. If you add credentials later, store them in Streamlit's **Advanced settings → Secrets**, and never commit them to GitHub.

## How to explain the analysis

- **Content performance:** median engagement rate by format. The P1 YouTube source supports a Shorts vs standard-video comparison and has no share metric, so rate = `(likes + comments) / views × 100`. Format is inferred from video title/hashtags. This is descriptive and may be biased by video age, channel size, and paid promotion.
- **Audience needs:** transparent English/Roman Urdu/Urdu-script keyword patterns surface common need categories. Unmatched comments are grouped with character n-gram TF-IDF and K-Means; those cluster keywords are leads to inspect, not validated customer segments.
- **P2 planning context:** P2 platform/campaign/budget summaries are synthetic. They can suggest hypotheses for later real campaign tests, but must not be reported as real brand performance or combined with P1 YouTube observations.
- **Unusual posts:** a robust median-absolute-deviation threshold flags engagement rates that differ from the dataset. A flag is a review prompt, not evidence of manipulation or success.
- **Recommendations:** suggest candidate content and message tests from observed patterns. Use controlled comparisons and sales or qualified inquiry outcomes to assess results.

## Project scope and next steps

This is a classroom-ready prototype rather than a production analytics system. The P1 export provides 525 YouTube videos and 8,641 comments; comments are available for 327 unique video IDs. The brief discusses cross-platform public social posts, while the supplied P1 files cover YouTube only. For the final report, state this scope clearly, manually review a sample of clusters, compare Shorts with Shorts and standard videos with standard videos, and describe missing data and limitations.

## Windows App Control error

If importing pandas fails with `DLL load failed while importing _compute` and says an Application Control policy blocked the file, Windows is blocking PyArrow's native library. Reinstalling packages in the same environment does not remove that policy. Ask your device administrator or university IT to approve the blocked dependency. The app can still run in Streamlit Community Cloud, which uses its own hosted environment.
