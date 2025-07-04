from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import json

keywords = [
    'n8n', 'stackai', 'workflow automation', 'gumloop', 'zapier',
    'all workflow automation keywords', 'use case in workflow automation',
    'n8n use cases', 'gumloop usecases', 'stackai use cases', 'zapier use cases',
    'workflow automation tools', 'workflow automation platforms',
    'automation workflow examples', 'workflow automation integrations',
    'workflow automation best practices', 'workflow automation trends',
    'workflow automation success stories', 'workflow automation case studies',
    'workflow automation challenges', 'workflow automation benefits'
]

def scrape_reddit(keywords, max_posts_per_kw=10):
    chrome_options = Options()
    chrome_options.add_argument("--incognito")
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(options=chrome_options)
    results = []

    try:
        for keyword in keywords:
            print(f"Searching Reddit for: {keyword}")
            search_url = f"https://www.reddit.com/search/?q={keyword.replace(' ', '+')}"
            driver.get(search_url)
            time.sleep(3)

            # Scroll to load more posts
            for _ in range(2):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)

            posts = driver.find_elements(By.CSS_SELECTOR, 'div[data-testid="post-container"]')
            for post in posts[:max_posts_per_kw]:
                try:
                    title_elem = post.find_element(By.CSS_SELECTOR, 'h3')
                    title = title_elem.text
                    url_elem = post.find_element(By.CSS_SELECTOR, 'a[data-click-id="body"]')
                    url = url_elem.get_attribute('href')

                    # Open post in a new tab to get content and comments
                    driver.execute_script("window.open('');")
                    driver.switch_to.window(driver.window_handles[1])
                    driver.get(url)
                    time.sleep(3)
                    try:
                        # Try to get the main post content
                        try:
                            content_elem = driver.find_element(By.CSS_SELECTOR, 'div[data-test-id="post-content"]')
                            content = content_elem.text
                        except Exception:
                            content = ""
                        # Get top 5 comments
                        comment_elems = driver.find_elements(By.CSS_SELECTOR, 'div[data-test-id="comment"]')
                        comments = []
                        for c in comment_elems[:5]:
                            try:
                                comment_text = c.text
                                comments.append(comment_text)
                            except Exception:
                                continue
                    except Exception:
                        content = ""
                        comments = []
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])
                    results.append({
                        'platform': 'Reddit',
                        'keyword': keyword,
                        'title': title,
                        'url': url,
                        'content': content,
                        'top_comments': comments
                    })
                except Exception:
                    continue
    finally:
        driver.quit()
    return results

if __name__ == "__main__":
    all_results = scrape_reddit(keywords)
    with open('reddit_selenium_scraped.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Scraping complete! All results stored in reddit_selenium_scraped.json.")
