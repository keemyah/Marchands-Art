import html
import json
import math
import os
import re
import sqlite3
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree as ET

import pandas as pd
import requests

# ============================================================
# Art Market Early Signal System
# ------------------------------------------------------------
# IMPORTANT
# - Do NOT save this file as "google.py".
# - Suggested filename: art_market_early_signal_system.py
# - UI mode:  streamlit run art_market_early_signal_system.py
# - CLI mode: python art_market_early_signal_system.py --cli
# ============================================================

# ---------- Optional Streamlit ----------
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    st = None
    STREAMLIT_AVAILABLE = False


def cache_data_fallback(*args, **kwargs) -> Callable:
    def decorator(func: Callable) -> Callable:
        return func
    return decorator


cache_data = st.cache_data if STREAMLIT_AVAILABLE else cache_data_fallback
if STREAMLIT_AVAILABLE:
    st.set_page_config(page_title="Art Market Early Signal System", layout="wide")

# ---------- Constants ----------
USER_AGENT = "Mozilla/5.0 (compatible; ArtMarketEarlySignal/2.0; +https://local.app)"
APP_DIR = Path("art_market_signal_data")
APP_DIR.mkdir(exist_ok=True)
DB_PATH = APP_DIR / "art_market_signal.db"
DEFAULT_EXPORT_ARTICLES = APP_DIR / "articles_latest.csv"
DEFAULT_EXPORT_SIGNALS = APP_DIR / "signals_latest.csv"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"

DEFAULT_ART_TERMS = [
    "auction", "art market", "gallery", "museum", "art fair", "exhibition",
    "Christie's", "Sotheby's", "Phillips", "Drouot", "collector", "biennale",
    "record sale", "hammer price", "blue-chip art", "primary market", "secondary market"
]

# Note: public RSS availability for third-party art sites changes over time.
# These source pages are treated as optional RSS/HTML sources and can be edited in the UI.
DEFAULT_EXTRA_SOURCES = {
    "ARTnews": "https://www.artnews.com/feed/",
    "Artnet News": "https://news.artnet.com/feed",
    "The Art Newspaper": "https://www.theartnewspaper.com/rss",
}

DEFAULT_TRUSTED_SOURCES = {
    "The Art Newspaper": 1.0,
    "ARTnews": 0.95,
    "Artnet News": 0.95,
    "Financial Times": 0.95,
    "Reuters": 0.95,
    "Bloomberg": 0.92,
    "The New York Times": 0.9,
    "The Wall Street Journal": 0.9,
    "Artforum": 0.9,
    "The Guardian": 0.85,
    "Le Monde": 0.85,
    "Apollo": 0.8,
    "Hyperallergic": 0.8,
    "Forbes": 0.7,
}

POSITIVE_TERMS = {
    "record", "surge", "booming", "rising", "breakthrough", "retrospective",
    "acquisition", "sold out", "award", "prize", "opening", "expansion",
    "successful", "milestone", "strong", "growth", "highlight", "major show"
}
NEGATIVE_TERMS = {
    "fraud", "lawsuit", "forgery", "scandal", "controversy", "collapse",
    "decline", "drop", "slump", "probe", "sanction", "ban", "closure",
    "theft", "fake", "dispute", "crisis"
}

MARKET_TERMS = {"auction", "sold", "hammer", "estimate", "record", "collector", "sale", "lot"}
INSTITUTIONAL_TERMS = {"museum", "retrospective", "biennale", "exhibition", "gallery", "curator", "foundation"}
RISK_TERMS = {"lawsuit", "fraud", "forgery", "sanction", "ban", "probe", "theft"}


@dataclass
class ParsedEntry:
    source_family: str
    title: str
    summary: str
    link: str
    source: str
    published: datetime | None


@dataclass
class NewsRow:
    entity: str
    source_family: str
    title: str
    source: str
    published: datetime | None
    link: str
    summary: str
    score_relevance: float
    score_source: float
    score_recency: float
    score_sentiment: float
    score_total: float
    category: str
    run_at: datetime


# ---------- Helpers ----------
def normalize_text(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def strip_title_suffix(title: str) -> Tuple[str, str]:
    if " - " not in title:
        return title, ""
    parts = title.rsplit(" - ", 1)
    if len(parts) != 2:
        return title, ""
    return parts[0].strip(), parts[1].strip()


def build_google_query(entity: str, art_terms: List[str], language: str, country: str, days: int) -> Tuple[str, str]:
    art_part = " OR ".join([f'"{t}"' for t in art_terms])
    entity_part = f'"{entity}"'
    when_part = f" when:{days}d" if days > 0 else ""
    query = f"({entity_part}) ({art_part}){when_part}"
    params = {
        "q": query,
        "hl": language,
        "gl": country,
        "ceid": f"{country}:{language.split('-')[0]}"
    }
    url = GOOGLE_NEWS_RSS + "?" + urllib.parse.urlencode(params)
    return query, url


def parse_rss_entries(xml_bytes: bytes, source_family: str) -> List[ParsedEntry]:
    root = ET.fromstring(xml_bytes)
    entries: List[ParsedEntry] = []

    for item in root.findall("./channel/item"):
        raw_title = normalize_text(item.findtext("title", default=""))
        title, source_from_title = strip_title_suffix(raw_title)
        summary = normalize_text(item.findtext("description", default=""))
        link = normalize_text(item.findtext("link", default=""))
        pub_date = parse_datetime(item.findtext("pubDate", default=None))

        source = source_from_title
        source_el = item.find("source")
        if source_el is not None and (source_el.text or "").strip():
            source = normalize_text(source_el.text)

        entries.append(
            ParsedEntry(
                source_family=source_family,
                title=title,
                summary=summary,
                link=link,
                source=source or source_family,
                published=pub_date,
            )
        )
    return entries


def source_weight(source: str, trusted: Dict[str, float]) -> float:
    if not source:
        return 0.5
    for name, weight in trusted.items():
        if name.lower() in source.lower():
            return weight
    return 0.6


def relevance_score(entity: str, title: str, summary: str, art_terms: List[str]) -> float:
    text = f"{title} {summary}".lower()
    score = 0.0

    entity_tokens = [t for t in re.split(r"\W+", entity.lower()) if t]
    if entity.lower() in text:
        score += 1.0
    elif entity_tokens and all(token in text for token in entity_tokens):
        score += 0.7

    hits = sum(1 for t in art_terms if t.lower() in text)
    score += min(hits * 0.18, 1.0)
    return min(score, 1.5)


def recency_score(published: datetime | None, half_life_days: float = 7.0) -> float:
    if published is None:
        return 0.4
    age_days = max((now_utc() - published).total_seconds() / 86400.0, 0)
    return math.exp(-math.log(2) * age_days / half_life_days)


def sentiment_score(title: str, summary: str) -> float:
    text = f"{title} {summary}".lower()
    pos = sum(1 for w in POSITIVE_TERMS if w in text)
    neg = sum(1 for w in NEGATIVE_TERMS if w in text)
    raw = pos - neg
    return max(min(0.5 + raw * 0.12, 1.0), 0.0)


def classify_article(title: str, summary: str) -> str:
    text = f"{title} {summary}".lower()
    if any(w in text for w in MARKET_TERMS):
        return "market"
    if any(w in text for w in INSTITUTIONAL_TERMS):
        return "institutional"
    if any(w in text for w in RISK_TERMS):
        return "risk"
    return "general"


def total_score(rel: float, src: float, rec: float, sent: float) -> float:
    return 0.40 * rel + 0.20 * src + 0.25 * rec + 0.15 * sent


def dedupe_key(entity: str, link: str, title: str) -> str:
    base = f"{entity.lower()}|{link.strip().lower()}|{title.strip().lower()}"
    return re.sub(r"\s+", " ", base)


# ---------- Storage ----------
def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at TEXT NOT NULL,
            entity TEXT NOT NULL,
            source_family TEXT NOT NULL,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            summary TEXT,
            link TEXT NOT NULL,
            published TEXT,
            category TEXT NOT NULL,
            relevance REAL NOT NULL,
            source_weight REAL NOT NULL,
            recency REAL NOT NULL,
            sentiment REAL NOT NULL,
            trend_score REAL NOT NULL,
            dedupe_key TEXT NOT NULL UNIQUE
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            as_of_date TEXT NOT NULL,
            entity TEXT NOT NULL,
            articles INTEGER NOT NULL,
            avg_trend_score REAL NOT NULL,
            peak_score REAL NOT NULL,
            attention_index REAL NOT NULL,
            article_delta_7d REAL,
            attention_delta_7d REAL,
            hype_signal INTEGER NOT NULL DEFAULT 0,
            UNIQUE(as_of_date, entity)
        )
        """
    )
    conn.commit()
    conn.close()


def save_articles(rows: List[NewsRow]) -> int:
    if not rows:
        return 0
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    inserted = 0

    for r in rows:
        try:
            cur.execute(
                """
                INSERT INTO articles (
                    run_at, entity, source_family, source, title, summary, link, published,
                    category, relevance, source_weight, recency, sentiment, trend_score, dedupe_key
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r.run_at.isoformat(),
                    r.entity,
                    r.source_family,
                    r.source,
                    r.title,
                    r.summary,
                    r.link,
                    r.published.isoformat() if r.published else None,
                    r.category,
                    r.score_relevance,
                    r.score_source,
                    r.score_recency,
                    r.score_sentiment,
                    r.score_total,
                    dedupe_key(r.entity, r.link, r.title),
                ),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            pass

    conn.commit()
    conn.close()
    return inserted


def load_articles_df(days: int = 30) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    cutoff = (now_utc() - timedelta(days=days)).isoformat()
    df = pd.read_sql_query(
        "SELECT * FROM articles WHERE run_at >= ? ORDER BY run_at DESC",
        conn,
        params=(cutoff,),
    )
    conn.close()
    if not df.empty:
        for col in ["run_at", "published"]:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def compute_daily_signals_from_articles(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "as_of_date", "entity", "articles", "avg_trend_score", "peak_score",
            "attention_index", "article_delta_7d", "attention_delta_7d", "hype_signal"
        ])

    work = df.copy()
    work["as_of_date"] = work["run_at"].dt.date.astype(str)

    daily = (
        work.groupby(["as_of_date", "entity"])
        .agg(
            articles=("title", "count"),
            avg_trend_score=("trend_score", "mean"),
            peak_score=("trend_score", "max"),
        )
        .reset_index()
        .sort_values(["entity", "as_of_date"])
    )
    daily["attention_index"] = daily["avg_trend_score"] * (1 + daily["articles"].apply(lambda x: math.log1p(float(x))))

    out_parts = []
    for entity, g in daily.groupby("entity", sort=False):
        g = g.copy().sort_values("as_of_date")
        g["articles_ma7"] = g["articles"].rolling(7, min_periods=2).mean()
        g["attention_ma7"] = g["attention_index"].rolling(7, min_periods=2).mean()
        g["article_delta_7d"] = g["articles"] - g["articles_ma7"]
        g["attention_delta_7d"] = g["attention_index"] - g["attention_ma7"]
        g["hype_signal"] = (
            (g["articles"] >= 3)
            & (g["articles"] > 2.0 * g["articles_ma7"].fillna(999999))
            | (g["attention_index"] > 1.75 * g["attention_ma7"].fillna(999999))
        ).astype(int)
        out_parts.append(g)

    out = pd.concat(out_parts, ignore_index=True)
    return out[[
        "as_of_date", "entity", "articles", "avg_trend_score", "peak_score",
        "attention_index", "article_delta_7d", "attention_delta_7d", "hype_signal"
    ]]


def upsert_daily_signals(df: pd.DataFrame) -> None:
    if df.empty:
        return
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for _, row in df.iterrows():
        cur.execute(
            """
            INSERT INTO daily_signals (
                as_of_date, entity, articles, avg_trend_score, peak_score,
                attention_index, article_delta_7d, attention_delta_7d, hype_signal
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(as_of_date, entity) DO UPDATE SET
                articles=excluded.articles,
                avg_trend_score=excluded.avg_trend_score,
                peak_score=excluded.peak_score,
                attention_index=excluded.attention_index,
                article_delta_7d=excluded.article_delta_7d,
                attention_delta_7d=excluded.attention_delta_7d,
                hype_signal=excluded.hype_signal
            """,
            (
                row["as_of_date"], row["entity"], int(row["articles"]), float(row["avg_trend_score"]),
                float(row["peak_score"]), float(row["attention_index"]),
                None if pd.isna(row["article_delta_7d"]) else float(row["article_delta_7d"]),
                None if pd.isna(row["attention_delta_7d"]) else float(row["attention_delta_7d"]),
                int(row["hype_signal"]),
            ),
        )
    conn.commit()
    conn.close()


def load_daily_signals_df(days: int = 90) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    cutoff = (now_utc() - timedelta(days=days)).date().isoformat()
    df = pd.read_sql_query(
        "SELECT * FROM daily_signals WHERE as_of_date >= ? ORDER BY as_of_date DESC, attention_index DESC",
        conn,
        params=(cutoff,),
    )
    conn.close()
    return df


# ---------- Fetchers ----------
@cache_data(ttl=1800, show_spinner=False)
def fetch_rss(url: str, source_family: str) -> List[ParsedEntry]:
    headers = {"User-Agent": USER_AGENT}
    response = requests.get(url, headers=headers, timeout=25)
    response.raise_for_status()
    return parse_rss_entries(response.content, source_family=source_family)


def fetch_google_news(entity: str, art_terms: List[str], language: str, country: str, days: int) -> List[ParsedEntry]:
    _, url = build_google_query(entity, art_terms, language, country, days)
    return fetch_rss(url, source_family="Google News")


def fetch_custom_source(url: str, source_family: str) -> List[ParsedEntry]:
    # Expects RSS XML. If the source stops publishing RSS, this source can be disabled in the UI.
    return fetch_rss(url, source_family=source_family)


def fetch_entity_news(entity: str, art_terms: List[str], trusted: Dict[str, float], language: str, country: str, days: int, extra_sources: Dict[str, str]) -> List[NewsRow]:
    parsed_entries: List[ParsedEntry] = []
    parsed_entries.extend(fetch_google_news(entity, art_terms, language, country, days))

    # Pull optional direct art-news sources and keep only items relevant to the entity.
    entity_tokens = [t for t in re.split(r"\W+", entity.lower()) if t]
    for source_family, url in extra_sources.items():
        try:
            entries = fetch_custom_source(url, source_family)
        except Exception:
            continue
        for e in entries:
            text = f"{e.title} {e.summary}".lower()
            if entity.lower() in text or (entity_tokens and all(tok in text for tok in entity_tokens)):
                parsed_entries.append(e)

    rows: List[NewsRow] = []
    run_at = now_utc()
    for entry in parsed_entries:
        rel = relevance_score(entity, entry.title, entry.summary, art_terms)
        src = source_weight(entry.source or entry.source_family, trusted)
        rec = recency_score(entry.published)
        sent = sentiment_score(entry.title, entry.summary)
        total = total_score(rel, src, rec, sent)
        category = classify_article(entry.title, entry.summary)

        rows.append(
            NewsRow(
                entity=entity,
                source_family=entry.source_family,
                title=entry.title,
                source=entry.source or entry.source_family,
                published=entry.published,
                link=entry.link,
                summary=entry.summary,
                score_relevance=rel,
                score_source=src,
                score_recency=rec,
                score_sentiment=sent,
                score_total=total,
                category=category,
                run_at=run_at,
            )
        )
    return rows


# ---------- API-style exports for ML ----------
def export_for_ml(days: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    articles = load_articles_df(days=days)
    signals = load_daily_signals_df(days=days)
    if not articles.empty:
        articles.to_csv(DEFAULT_EXPORT_ARTICLES, index=False)
    if not signals.empty:
        signals.to_csv(DEFAULT_EXPORT_SIGNALS, index=False)
    return articles, signals


def latest_signal_features() -> pd.DataFrame:
    signals = load_daily_signals_df(days=365)
    if signals.empty:
        return signals
    signals = signals.sort_values(["entity", "as_of_date"]).groupby("entity", as_index=False).tail(1)
    return signals[[
        "entity", "articles", "avg_trend_score", "peak_score", "attention_index",
        "article_delta_7d", "attention_delta_7d", "hype_signal"
    ]].sort_values("attention_index", ascending=False)


# ---------- Diagnostics ----------
def _run_self_tests() -> None:
    assert normalize_text("A&nbsp;<b>test</b>") == "A test"
    title, source = strip_title_suffix("Banksy show opens - The Art Newspaper")
    assert title == "Banksy show opens"
    assert source == "The Art Newspaper"
    assert parse_datetime("Mon, 10 Mar 2025 12:00:00 GMT") is not None

    sample_xml = b"""
    <rss version="2.0"><channel>
      <item>
        <title>Banksy record sale - Reuters</title>
        <link>https://example.com/a</link>
        <description><![CDATA[Big auction result at Christie's]]></description>
        <pubDate>Mon, 10 Mar 2025 12:00:00 GMT</pubDate>
      </item>
    </channel></rss>
    """
    entries = parse_rss_entries(sample_xml, source_family="Google News")
    assert len(entries) == 1
    assert entries[0].source == "Reuters"
    assert entries[0].title == "Banksy record sale"


_run_self_tests()
init_db()


# ---------- CLI ----------
def run_cli() -> None:
    print("Art Market Early Signal System - CLI mode\n")
    entities = ["Banksy", "Yayoi Kusama", "Amoako Boafo", "Art Basel", "Christie's"]
    rows: List[NewsRow] = []
    for entity in entities:
        print(f"Fetching: {entity}")
        try:
            rows.extend(fetch_entity_news(entity, DEFAULT_ART_TERMS, DEFAULT_TRUSTED_SOURCES, "en-US", "US", 14, DEFAULT_EXTRA_SOURCES))
        except Exception as e:
            print(f"Error for {entity}: {e}")

    inserted = save_articles(rows)
    print(f"Inserted {inserted} new articles into SQLite.")

    articles = load_articles_df(days=30)
    daily = compute_daily_signals_from_articles(articles)
    upsert_daily_signals(daily)
    latest = latest_signal_features()

    if latest.empty:
        print("No signals available.")
        return

    print("\nLatest entity signals:\n")
    print(latest.head(15).to_string(index=False))


# ---------- Streamlit UI ----------
def run_streamlit() -> None:
    st.title("Art Market Early Signal System")
    st.caption("Veille avancée : Google News + sources art, base SQLite, historique, détection de hype et exports ML.")

    with st.sidebar:
        st.header("Paramètres")
        entities_text = st.text_area(
            "Artistes / thèmes à suivre",
            value="Banksy\nYayoi Kusama\nAmoako Boafo\nJulie Mehretu\nArt Basel\nChristie's",
            height=180,
        )
        custom_terms = st.text_input(
            "Mots-clés marché de l'art",
            value=", ".join(DEFAULT_ART_TERMS),
        )
        language = st.selectbox("Langue", ["fr-FR", "en-US", "en-GB"], index=1)
        country = st.selectbox("Pays / édition", ["FR", "US", "GB"], index=1)
        days = st.slider("Fenêtre de recherche (jours)", min_value=1, max_value=30, value=14)
        history_days = st.slider("Historique affiché (jours)", min_value=7, max_value=365, value=90)
        max_entities = st.slider("Nombre max d'entités", min_value=1, max_value=50, value=10)
        source_json = st.text_area(
            "Sources RSS additionnelles (JSON)",
            value=json.dumps(DEFAULT_EXTRA_SOURCES, ensure_ascii=False, indent=2),
            height=180,
        )
        run_button = st.button("Lancer la collecte", type="primary")

    st.markdown(
        "Le système collecte des articles, les stocke dans **SQLite**, calcule un **trend score**, "
        "puis détecte les **explosions médiatiques** (hype signals) par entité."
    )

    entities = [e.strip() for e in entities_text.splitlines() if e.strip()][:max_entities]
    art_terms = [t.strip() for t in custom_terms.split(",") if t.strip()]
    try:
        extra_sources = json.loads(source_json) if source_json.strip() else {}
        if not isinstance(extra_sources, dict):
            extra_sources = {}
    except json.JSONDecodeError:
        st.warning("Le JSON des sources est invalide. Les sources additionnelles seront ignorées.")
        extra_sources = {}

    if run_button:
        if not entities:
            st.warning("Ajoute au moins une entité à suivre.")
            st.stop()

        rows: List[NewsRow] = []
        progress = st.progress(0)
        status = st.empty()

        for i, entity in enumerate(entities, start=1):
            status.write(f"Collecte pour : {entity}")
            try:
                rows.extend(fetch_entity_news(entity, art_terms, DEFAULT_TRUSTED_SOURCES, language, country, days, extra_sources))
            except Exception as e:
                st.error(f"Erreur pour {entity}: {e}")
            progress.progress(i / len(entities))

        inserted = save_articles(rows)
        st.success(f"Collecte terminée. {inserted} nouveaux articles enregistrés dans SQLite.")

        articles_df = load_articles_df(days=history_days)
        daily_df = compute_daily_signals_from_articles(articles_df)
        upsert_daily_signals(daily_df)

    articles_df = load_articles_df(days=history_days)
    daily_df = load_daily_signals_df(days=history_days)
    latest_df = latest_signal_features()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Articles en base", 0 if articles_df.empty else len(articles_df))
    c2.metric("Entités avec signal", 0 if latest_df.empty else latest_df["entity"].nunique())
    c3.metric("Hype signals actifs", 0 if latest_df.empty else int(latest_df["hype_signal"].sum()))
    c4.metric("Attention moyenne", 0 if latest_df.empty else round(latest_df["attention_index"].mean(), 3))

    st.subheader("Derniers signaux par entité")
    if latest_df.empty:
        st.info("Aucun signal en base pour le moment. Lance une collecte.")
    else:
        st.dataframe(latest_df, use_container_width=True, hide_index=True)

    st.subheader("Historique d'une entité")
    if not daily_df.empty:
        entity_options = latest_df["entity"].tolist() if not latest_df.empty else sorted(daily_df["entity"].unique().tolist())
        chosen = st.selectbox("Entité", options=entity_options)
        chosen_df = daily_df[daily_df["entity"] == chosen].copy().sort_values("as_of_date")
        chosen_df["as_of_date"] = pd.to_datetime(chosen_df["as_of_date"])

        chart_df = chosen_df[["as_of_date", "attention_index", "articles"]].set_index("as_of_date")
        st.line_chart(chart_df)

        st.write("Historique détaillé")
        st.dataframe(chosen_df.sort_values("as_of_date", ascending=False), use_container_width=True, hide_index=True)

        st.write("Articles récents liés à cette entité")
        entity_articles = articles_df[articles_df["entity"] == chosen].copy().sort_values(["run_at", "trend_score"], ascending=[False, False])
        st.dataframe(
            entity_articles[["source_family", "source", "title", "category", "published", "trend_score", "link"]],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Exports pour modèle ML")
    articles_export, signals_export = export_for_ml(days=history_days)
    if not articles_export.empty:
        st.download_button(
            "Télécharger articles_latest.csv",
            DEFAULT_EXPORT_ARTICLES.read_bytes(),
            file_name="articles_latest.csv",
            mime="text/csv",
        )
    if not signals_export.empty:
        st.download_button(
            "Télécharger signals_latest.csv",
            DEFAULT_EXPORT_SIGNALS.read_bytes(),
            file_name="signals_latest.csv",
            mime="text/csv",
        )

    st.code(
        """
# Exemple d'usage ML
import pandas as pd
signals = pd.read_csv('art_market_signal_data/signals_latest.csv')
# merge on artist/entity name with your art-pricing dataset
        """.strip(),
        language="python",
    )


if __name__ == "__main__":
    import sys
    if ("--cli" in sys.argv) or (not STREAMLIT_AVAILABLE):
        run_cli()
    else:
        run_streamlit()

