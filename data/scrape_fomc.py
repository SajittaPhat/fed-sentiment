"""
scrape_fomc.py
--------------
Downloads FOMC post-meeting statements from federalreserve.gov.

TWO URL patterns confirmed from live site:
  2000-2008:  /boarddocs/press/monetary/YYYY/YYYYMMDD/default.htm
  2009-now:   /newsevents/pressreleases/monetaryYYYYMMDDa.htm
"""

import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent / "fomc_statements2002-05-07-2008-12-16.csv"
MIN_CHARS   = 150

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection":      "keep-alive",
}

FOMC_URLS = [
    # ── 2000-2008: /boarddocs/press/monetary/YYYY/YYYYMMDD/default.htm ───────
    # 2000
    "https://www.federalreserve.gov/boarddocs/press/monetary/2000/20000202/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2000/20000321/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2000/20000516/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2000/20000628/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2000/20000822/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2000/20001003/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2000/20001115/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2000/20001219/default.htm",
    # 2001
    "https://www.federalreserve.gov/boarddocs/press/monetary/2001/20010103/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2001/20010131/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2001/20010320/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2001/20010418/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2001/20010515/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2001/20010627/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2001/20010821/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2001/20010917/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2001/20011002/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2001/20011106/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2001/20011211/default.htm",
    # 2002
    "https://www.federalreserve.gov/boarddocs/press/monetary/2002/20020130/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2002/20020319/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2002/20020507/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2002/20020626/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2002/20020813/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2002/20020924/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2002/20021106/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2002/20021210/default.htm",
    # 2003
    "https://www.federalreserve.gov/boarddocs/press/monetary/2003/20030129/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2003/20030318/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2003/20030506/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2003/20030625/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2003/20030812/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2003/20030916/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2003/20031028/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2003/20031209/default.htm",
    # 2004
    "https://www.federalreserve.gov/boarddocs/press/monetary/2004/20040128/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2004/20040316/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2004/20040504/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2004/20040630/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2004/20040810/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2004/20040921/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2004/20041103/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2004/20041214/default.htm",
    # 2005
    "https://www.federalreserve.gov/boarddocs/press/monetary/2005/20050202/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2005/20050322/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2005/20050503/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2005/20050630/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2005/20050809/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2005/20050920/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2005/20051101/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2005/20051213/default.htm",
    # 2006
    "https://www.federalreserve.gov/boarddocs/press/monetary/2006/20060131/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2006/20060328/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2006/20060510/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2006/20060629/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2006/20060808/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2006/20060920/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2006/20061025/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2006/20061212/default.htm",
    # 2007
    "https://www.federalreserve.gov/boarddocs/press/monetary/2007/20070131/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2007/20070321/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2007/20070509/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2007/20070628/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2007/20070807/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2007/20070918/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2007/20071031/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2007/20071211/default.htm",
    # 2008
    "https://www.federalreserve.gov/boarddocs/press/monetary/2008/20080122/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2008/20080130/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2008/20080318/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2008/20080430/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2008/20080625/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2008/20080805/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2008/20080916/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2008/20081008/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2008/20081029/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/2008/20081216/default.htm",
    # ── 2009-2024: /newsevents/pressreleases/monetaryYYYYMMDDa.htm ───────────
    # 2009
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20090128a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20090318a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20090429a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20090624a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20090812a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20090923a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20091104a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20091216a.htm",
    # 2010
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20100127a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20100316a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20100428a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20100623a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20100810a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20100921a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20101103a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20101214a.htm",
    # 2011
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20110126a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20110315a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20110427a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20110622a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20110809a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20110921a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20111102a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20111213a.htm",
    # 2012
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20120125a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20120313a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20120425a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20120620a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20120801a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20120913a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20121024a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20121212a.htm",
    # 2013
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20130130a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20130320a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20130501a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20130619a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20130731a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20130918a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20131030a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20131218a.htm",
    # 2014
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20140129a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20140319a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20140430a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20140618a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20140730a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20140917a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20141029a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20141217a.htm",
    # 2015
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20150128a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20150318a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20150429a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20150617a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20150729a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20150917a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20151028a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20151216a.htm",
    # 2016
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20160127a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20160316a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20160427a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20160615a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20160727a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20160921a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20161102a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20161214a.htm",
    # 2017
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20170201a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20170315a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20170503a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20170614a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20170726a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20170920a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20171101a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20171213a.htm",
    # 2018
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20180131a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20180321a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20180502a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20180613a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20180801a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20180926a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20181108a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20181219a.htm",
    # 2019
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20190130a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20190320a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20190501a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20190619a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20190731a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20190918a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20191030a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20191211a.htm",
    # 2020
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20200129a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20200303a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20200315a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20200429a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20200610a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20200729a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20200916a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20201105a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20201216a.htm",
    # 2021
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20210127a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20210317a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20210428a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20210616a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20210728a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20210922a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20211103a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20211215a.htm",
    # 2022
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20220126a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20220316a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20220504a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20220615a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20220727a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20220921a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20221102a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20221214a.htm",
    # 2023
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20230201a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20230322a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20230503a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20230614a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20230726a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20230920a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20231101a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20231213a.htm",
    # 2024
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240131a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240320a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240501a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240612a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240731a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240918a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20241107a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20241218a.htm",
]


def date_from_url(url):
    m = re.search(r"(\d{8})", url)
    if m:
        r = m.group(1)
        return f"{r[:4]}-{r[4:6]}-{r[6:8]}"
    return None


def extract_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for el in soup.find_all(["script", "style", "nav", "header", "footer"]):
        el.decompose()

    # Modern layout (2014+)
    for cls in ["col-xs-12 col-sm-8 col-md-8", "col-sm-8", "col-md-8"]:
        tag = soup.find("div", class_=cls)
        if tag:
            t = re.sub(r"\s+", " ", tag.get_text(separator=" ")).strip()
            if len(t) >= MIN_CHARS:
                return t

    # Mid-era divs (2009-2013)
    for div_id in ["leftText", "content", "article"]:
        tag = soup.find("div", id=div_id)
        if tag:
            t = re.sub(r"\s+", " ", tag.get_text(separator=" ")).strip()
            if len(t) >= MIN_CHARS:
                return t

    # Legacy boarddocs table layout (2000-2008) — pick largest <td>
    tds = soup.find_all("td")
    if tds:
        best = max(tds, key=lambda td: len(td.get_text()))
        t = re.sub(r"\s+", " ", best.get_text(separator=" ")).strip()
        if len(t) >= MIN_CHARS:
            return t

    # Paragraph fallback
    paras = [re.sub(r"\s+", " ", p.get_text()).strip()
             for p in soup.find_all("p")
             if len(p.get_text(strip=True)) > 40]
    if paras:
        t = " ".join(paras)
        if len(t) >= MIN_CHARS:
            return t

    return None


def scrape_all():
    print(f"Scraping {len(FOMC_URLS)} FOMC statements...\n")
    records, skipped = [], []

    for i, url in enumerate(FOMC_URLS):
        date_str = date_from_url(url)
        if not date_str:
            continue

        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            if resp.status_code == 404:
                skipped.append((date_str, "404"))
                print(f"  [{i+1:3d}/{len(FOMC_URLS)}] {date_str}  SKIPPED [404]")
            elif resp.status_code != 200:
                skipped.append((date_str, f"http_{resp.status_code}"))
                print(f"  [{i+1:3d}/{len(FOMC_URLS)}] {date_str}  SKIPPED [HTTP {resp.status_code}]")
            else:
                text = extract_text(resp.text)
                if text:
                    records.append({"date": date_str, "text": text, "url": url})
                    print(f"  [{i+1:3d}/{len(FOMC_URLS)}] {date_str}  OK  ({len(text):,} chars)")
                else:
                    skipped.append((date_str, "empty"))
                    print(f"  [{i+1:3d}/{len(FOMC_URLS)}] {date_str}  SKIPPED [no text]")

        except requests.exceptions.SSLError:
            skipped.append((date_str, "ssl"))
            print(f"  [{i+1:3d}/{len(FOMC_URLS)}] {date_str}  SKIPPED [SSL error]")
        except requests.exceptions.ConnectionError:
            skipped.append((date_str, "connection"))
            print(f"  [{i+1:3d}/{len(FOMC_URLS)}] {date_str}  SKIPPED [connection error]")
        except Exception as e:
            skipped.append((date_str, str(e)[:40]))
            print(f"  [{i+1:3d}/{len(FOMC_URLS)}] {date_str}  SKIPPED [{type(e).__name__}]")

        time.sleep(0.4)

    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'='*50}")
    print(f"Saved   : {len(records)} statements → {OUTPUT_PATH}")
    print(f"Skipped : {len(skipped)}")
    return df


if __name__ == "__main__":
    scrape_all()