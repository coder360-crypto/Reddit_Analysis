import os
import praw
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import json
from datetime import datetime

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

class RedditIndustryUseCaseAnalyzer:
    def __init__(self, client_id=None, client_secret=None, user_agent=None, username=None, password=None):
        self.reddit = None
        if client_id and client_secret and user_agent and username and password:
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
                username=username,
                password=password
            )
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.scraped_data = []
        self.processed_texts = []
        self.processed_data = []

    def scrape_subreddit_posts(self, subreddit_name, keywords=None, limit=50):
        if not self.reddit:
            print("Reddit API not initialized.")
            return
        print(f"Scraping r/{subreddit_name}...")
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts = subreddit.hot(limit=limit)
            for post in posts:
                post_text = (post.title + " " + (post.selftext or "")).lower()
                if keywords and not any(keyword.lower() in post_text for keyword in keywords):
                    continue
                post_data = {
                    'id': post.id,
                    'title': post.title,
                    'selftext': post.selftext,
                    'subreddit': subreddit_name,
                    'created_utc': datetime.fromtimestamp(post.created_utc).isoformat(),
                    'comments': []
                }
                post.comments.replace_more(limit=0)
                for comment in post.comments.list()[:10]:
                    if hasattr(comment, 'body') and comment.body != '[deleted]':
                        post_data['comments'].append(comment.body)
                self.scraped_data.append(post_data)
        except Exception as e:
            print(f"Error scraping r/{subreddit_name}: {str(e)}")

    def search_reddit(self, query, limit=30):
        if not self.reddit:
            print("Reddit API not initialized.")
            return
        print(f"Searching Reddit for: '{query}'...")
        try:
            search_results = self.reddit.subreddit('all').search(query, limit=limit, sort='relevance')
            for post in search_results:
                post_data = {
                    'id': post.id,
                    'title': post.title,
                    'selftext': post.selftext,
                    'subreddit': post.subreddit.display_name,
                    'created_utc': datetime.fromtimestamp(post.created_utc).isoformat(),
                    'comments': []
                }
                post.comments.replace_more(limit=0)
                for comment in post.comments.list()[:5]:
                    if hasattr(comment, 'body') and comment.body != '[deleted]':
                        post_data['comments'].append(comment.body)
                self.scraped_data.append(post_data)
        except Exception as e:
            print(f"Error searching for '{query}': {str(e)}")

    def save_scraped_data(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_data, f, ensure_ascii=False, indent=2)
        print(f"Scraped data saved to {filename}")

    def load_scraped_data(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            self.scraped_data = json.load(f)
        print(f"Loaded {len(self.scraped_data)} records from {filename}")

    def preprocess_text(self, text):
        if not text:
            return ""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower().strip()
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)

    def classify_industry(self, text, industry_keywords):
        found = set()
        for industry, keywords in industry_keywords.items():
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw.lower()) + r'\b', text):
                    found.add(industry)
        return found

    def process_all_data(self, industry_keywords):
        print("Processing scraped data...")
        processed_data = []
        for post in self.scraped_data:
            full_text = post['title'] + " " + (post['selftext'] or "")
            comments_text = " ".join(post['comments'])
            full_text_with_comments = full_text + " " + comments_text
            processed_text = self.preprocess_text(full_text_with_comments)
            if processed_text:
                industries = self.classify_industry(processed_text, industry_keywords)
                processed_item = {
                    'id': post['id'],
                    'subreddit': post['subreddit'],
                    'title': post['title'],
                    'original_text': full_text_with_comments,
                    'processed_text': processed_text,
                    'created_utc': post['created_utc'],
                    'industries': list(industries)
                }
                processed_data.append(processed_item)
                self.processed_texts.append(processed_text)
        self.processed_data = processed_data
        print(f"Processed {len(processed_data)} posts")

    def count_industries(self):
        industry_counter = Counter()
        for item in self.processed_data:
            for industry in item['industries']:
                industry_counter[industry] += 1
        return industry_counter

    def visualize_industry_counts(self, industry_counter):
        if not industry_counter:
            print("No industry data to visualize.")
            return
        industries, counts = zip(*industry_counter.most_common())
        plt.figure(figsize=(12, 6))
        plt.bar(industries, counts, color='skyblue')
        plt.xlabel('Industry')
        plt.ylabel('Number of Use Cases')
        plt.title('Workflow Automation Use Cases by Industry')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Pie chart for top 8 industries
        top_industries = industry_counter.most_common(8)
        labels, sizes = zip(*top_industries)
        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title('Top Industries in Workflow Automation Use Cases')
        plt.show()

    def save_processed_data(self, filename_prefix='reddit_industry_analysis'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        df = pd.DataFrame(self.processed_data)
        df.to_csv(f'{filename_prefix}_processed_{timestamp}.csv', index=False, encoding='utf-8')
        print(f"Processed data saved to {filename_prefix}_processed_{timestamp}.csv")

def main():
    json_filename = 'reddit_scraped.json'

    industry_keywords = {
        "Healthcare": ["healthcare", "hospital", "patient", "medical", "clinic", "ehr", "pharmacy"],
        "Finance": ["finance", "bank", "banking", "insurance", "fintech", "accounting", "investment", "trading"],
        "Education": ["education", "school", "university", "student", "teacher", "edtech", "classroom"],
        "E-commerce": ["ecommerce", "shopify", "magento", "store", "retail", "online store"],
        "Marketing": ["marketing", "crm", "campaign", "lead", "advertising", "salesforce", "hubspot"],
        "Manufacturing": ["manufacturing", "factory", "supply chain", "production", "logistics", "inventory"],
        "Human Resources": ["hr", "human resources", "recruitment", "hiring", "payroll", "onboarding"],
        "IT/Software": ["it", "software", "devops", "cloud", "server", "saas", "api", "integration"],
        "Real Estate": ["real estate", "property", "broker", "agent", "mortgage", "listing"],
        "Legal": ["legal", "law", "contract", "compliance", "attorney", "lawyer"],
        "Travel": ["travel", "booking", "hotel", "flight", "tourism", "trip"],
        "Government": ["government", "public sector", "municipal", "civic", "agency"],
        "Energy": ["energy", "utility", "power", "oil", "gas", "solar", "renewable"],
        "Other": ["general", "misc", "other"]
    }

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

    subreddits = [
        'n8n_io', 'workflowautomation', 'zapier', 'nocode', 'automation',
        'Productivity', 'stackai', 'gumloop', 'SaaS', 'Entrepreneur'
    ]

    analyzer = RedditIndustryUseCaseAnalyzer()

    if os.path.isfile(json_filename):
        print(f"Found {json_filename}, loading and analyzing...")
        analyzer.load_scraped_data(json_filename)
    else:
        print(f"{json_filename} not found. Scraping new data from Reddit...")
        # Get credentials securely
        CLIENT_ID = 
        CLIENT_SECRET = 
        USER_AGENT = 
        USERNAME = 
        PASSWORD = 
        analyzer = RedditIndustryUseCaseAnalyzer(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent=USER_AGENT,
            username=USERNAME,
            password=PASSWORD
        )
        for subreddit in subreddits:
            analyzer.scrape_subreddit_posts(subreddit, keywords=keywords, limit=100)
        for keyword in keywords:
            analyzer.search_reddit(keyword, limit=100)
        analyzer.save_scraped_data(json_filename)

    analyzer.process_all_data(industry_keywords)
    industry_counter = analyzer.count_industries()
    print("\nIndustry use case counts:")
    for industry, count in industry_counter.most_common():
        print(f"{industry}: {count}")
    analyzer.visualize_industry_counts(industry_counter)
    analyzer.save_processed_data('reddit_industry_analysis')

if __name__ == "__main__":
    main()
