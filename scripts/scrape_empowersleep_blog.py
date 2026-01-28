#!/usr/bin/env python3
"""
scripts/scrape_empowersleep_blog.py
====================================

Scrapes EmpowerSleep blog articles and saves them as JSONL for RAG ingestion.

Usage:
    python scripts/scrape_empowersleep_blog.py

Output:
    data/blog_docs.jsonl - One JSON object per line with {title, url, text}

Dependencies:
    pip install requests beautifulsoup4
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, NavigableString

# =============================================================================
# CONFIGURATION
# =============================================================================

BLOG_INDEX_URL = "https://www.empowersleep.com/blog"
ARTICLE_URL_PATTERN = "/articles/"  # URLs containing this are articles

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "blog_docs.jsonl"

# Request settings
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds, doubles each retry
DELAY_BETWEEN_REQUESTS = 1  # seconds, be polite to the server

# Content filtering
MIN_TEXT_LENGTH = 500  # Skip articles with less text than this


# =============================================================================
# HTTP UTILITIES
# =============================================================================

def fetch_url(url: str, retries: int = MAX_RETRIES) -> Optional[str]:
    """
    Fetch a URL with retries and exponential backoff.

    Args:
        url: The URL to fetch
        retries: Number of retries remaining

    Returns:
        HTML content as string, or None if all retries failed
    """
    headers = {"User-Agent": USER_AGENT}
    backoff = RETRY_BACKOFF

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            if attempt < retries - 1:
                print(f"  Warning: Request failed ({e}), retrying in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2  # Exponential backoff
            else:
                print(f"  Error: Failed to fetch {url} after {retries} attempts: {e}")
                return None

    return None


# =============================================================================
# URL EXTRACTION
# =============================================================================

def extract_article_urls(blog_index_html: str, base_url: str) -> List[str]:
    """
    Extract all unique article URLs from the blog index page.

    Args:
        blog_index_html: HTML content of the blog index page
        base_url: Base URL for resolving relative links

    Returns:
        List of unique, absolute article URLs
    """
    soup = BeautifulSoup(blog_index_html, "html.parser")

    urls: Set[str] = set()

    # Find all links on the page
    for link in soup.find_all("a", href=True):
        href = link["href"]

        # Convert to absolute URL
        absolute_url = urljoin(base_url, href)

        # Check if it's an article URL
        if ARTICLE_URL_PATTERN in absolute_url:
            # Normalize: remove fragments and trailing slashes
            parsed = urlparse(absolute_url)
            normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
            urls.add(normalized)

    # Sort for consistent ordering
    return sorted(urls)


# =============================================================================
# CONTENT EXTRACTION
# =============================================================================

def extract_title(soup: BeautifulSoup) -> str:
    """
    Extract the best title from the page.

    Priority:
    1. First <h1> tag
    2. <title> tag (cleaned of site suffix)
    3. Fallback to "Untitled"
    """
    # Try h1 first
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)

    # Fallback to <title>
    title_tag = soup.find("title")
    if title_tag and title_tag.get_text(strip=True):
        title = title_tag.get_text(strip=True)
        # Remove common site suffixes like " | EmpowerSleep" or " - EmpowerSleep"
        title = re.split(r"\s*[|\-–—]\s*", title)[0].strip()
        return title

    return "Untitled"


def extract_article_text(soup: BeautifulSoup) -> str:
    """
    Extract clean, readable text from the article.

    Strategy:
    1. Remove known noise elements (nav, footer, scripts, etc.)
    2. Try to find the main content area
    3. Extract text from paragraphs, lists, and headings
    4. Preserve section structure with headings for better chunking
    """
    # Remove noise elements that we never want
    noise_selectors = [
        "nav", "footer", "header", "aside",
        "script", "style", "noscript",
        ".nav", ".navbar", ".menu", ".sidebar",
        ".footer", ".header", ".navigation",
        ".cookie", ".popup", ".modal", ".banner",
        ".social", ".share", ".comments", ".related",
        "[role='navigation']", "[role='banner']",
    ]

    for selector in noise_selectors:
        for element in soup.select(selector):
            element.decompose()

    # Try to find main content area (common patterns)
    main_content = None
    content_selectors = [
        "article",
        "main",
        "[role='main']",
        ".article-content",
        ".post-content",
        ".entry-content",
        ".content",
        ".article-body",
        ".blog-post",
    ]

    for selector in content_selectors:
        main_content = soup.select_one(selector)
        if main_content:
            break

    # Fallback to body if no main content found
    if not main_content:
        main_content = soup.find("body") or soup

    # Extract text with structure
    text_parts: List[str] = []

    # Content tags we want to extract (in order of appearance)
    content_tags = ["h1", "h2", "h3", "h4", "p", "li", "blockquote"]

    for element in main_content.find_all(content_tags):
        # Skip if inside a noise element (double-check)
        if element.find_parent(["nav", "footer", "header", "aside"]):
            continue

        text = element.get_text(strip=True)

        if not text:
            continue

        # Format headings distinctly for better chunking
        if element.name in ["h1", "h2", "h3", "h4"]:
            # Add newlines around headings
            text_parts.append(f"\n## {text}\n")
        elif element.name == "li":
            # Format list items with bullet
            text_parts.append(f"- {text}")
        elif element.name == "blockquote":
            # Format blockquotes
            text_parts.append(f"> {text}")
        else:
            # Regular paragraph
            text_parts.append(text)

    # Join with newlines, collapse multiple newlines
    full_text = "\n".join(text_parts)
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)

    return full_text.strip()


def scrape_article(url: str) -> Optional[Dict[str, str]]:
    """
    Scrape a single article and return structured data.

    Args:
        url: The article URL

    Returns:
        Dict with title, url, text - or None if scraping failed
    """
    html = fetch_url(url)
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")

    title = extract_title(soup)
    text = extract_article_text(soup)

    # Skip articles with very little content
    if len(text) < MIN_TEXT_LENGTH:
        print(f"  Skipping (too short: {len(text)} chars): {title}")
        return None

    return {
        "title": title,
        "url": url,
        "text": text,
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def scrape_blog() -> List[Dict[str, str]]:
    """
    Main scraping pipeline.

    Returns:
        List of article dicts with title, url, text
    """
    print(f"Fetching blog index: {BLOG_INDEX_URL}")

    # Step 1: Fetch the blog index page
    index_html = fetch_url(BLOG_INDEX_URL)
    if not index_html:
        print("Error: Could not fetch blog index page")
        return []

    # Step 2: Extract article URLs
    article_urls = extract_article_urls(index_html, BLOG_INDEX_URL)
    print(f"Found {len(article_urls)} unique article URLs")

    if not article_urls:
        print("Warning: No article URLs found")
        return []

    # Step 3: Scrape each article
    articles: List[Dict[str, str]] = []

    for i, url in enumerate(article_urls, 1):
        print(f"Fetching {i}/{len(article_urls)}: {url}")

        article = scrape_article(url)
        if article:
            articles.append(article)
            print(f"  OK: {article['title']} ({len(article['text'])} chars)")

        # Be polite: delay between requests
        if i < len(article_urls):
            time.sleep(DELAY_BETWEEN_REQUESTS)

    print(f"\nSuccessfully scraped {len(articles)}/{len(article_urls)} articles")
    return articles


def save_jsonl(articles: List[Dict[str, str]], output_path: Path) -> None:
    """
    Save articles to a JSONL file (one JSON object per line).

    Args:
        articles: List of article dicts
        output_path: Path to output file
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for article in articles:
            json_line = json.dumps(article, ensure_ascii=False)
            f.write(json_line + "\n")

    print(f"Saved {len(articles)} articles to {output_path}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("EmpowerSleep Blog Scraper")
    print("=" * 60)

    # Scrape all articles
    articles = scrape_blog()

    if not articles:
        print("No articles scraped. Exiting.")
        return

    # Save to JSONL
    save_jsonl(articles, OUTPUT_PATH)

    # Summary
    total_chars = sum(len(a["text"]) for a in articles)
    print(f"\nSummary:")
    print(f"  Articles: {len(articles)}")
    print(f"  Total text: {total_chars:,} characters")
    print(f"  Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
