import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import re
from datetime import datetime
import numpy as np
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
import spacy

# Statistical analysis
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RedditDataVisualizer:
    def __init__(self, data_file='reddit_scraped.json'):
        self.data_file = data_file
        self.scraped_data = []
        self.processed_data = []
        self.df = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Enhanced stop words list - removing generic and common words
        self.stop_words.update([
            # Original workflow-related terms
            'workflow', 'automation', 'n8n', 'zapier', 'use', 'case', 'tool', 'platform',
            
            # Generic action words
            'work', 'make', 'get', 'take', 'give', 'put', 'set', 'run', 'start', 'stop',
            'create', 'build', 'add', 'remove', 'delete', 'update', 'change', 'move',
            'help', 'find', 'look', 'see', 'show', 'tell', 'know', 'think', 'want',
            'need', 'try', 'go', 'come', 'turn', 'keep', 'let', 'call', 'ask',
            
            # Generic nouns
            'time', 'people', 'person', 'thing', 'way', 'day', 'year', 'week', 'month',
            'place', 'part', 'side', 'hand', 'eye', 'head', 'face', 'fact', 'end',
            'point', 'group', 'number', 'area', 'money', 'story', 'lot', 'right',
            'study', 'book', 'word', 'issue', 'service', 'user', 'customer', 'client',
            
            # Generic adjectives
            'good', 'new', 'first', 'last', 'long', 'great', 'little', 'own', 'old',
            'right', 'big', 'high', 'different', 'small', 'large', 'next', 'early',
            'important', 'few', 'public', 'bad', 'same', 'able', 'better', 'best',
            'real', 'sure', 'local', 'certain', 'free', 'full', 'available', 'current',
            
            # Generic pronouns and determiners
            'much', 'many', 'most', 'more', 'less', 'few', 'little', 'every', 'each',
            'some', 'any', 'all', 'both', 'either', 'neither', 'other', 'another',
            'such', 'what', 'which', 'who', 'where', 'when', 'why', 'how',
            
            # Generic connecting words
            'well', 'also', 'just', 'even', 'still', 'really', 'quite', 'pretty',
            'very', 'too', 'so', 'then', 'now', 'here', 'there', 'back', 'only',
            'actually', 'probably', 'maybe', 'perhaps', 'definitely', 'certainly',
            
            # Generic business/work terms
            'company', 'business', 'organization', 'team', 'project', 'job', 'task',
            'process', 'system', 'solution', 'problem', 'issue', 'question', 'answer',
            'result', 'experience', 'example', 'case', 'situation', 'condition',
            'requirement', 'feature', 'function', 'option', 'choice', 'decision',
            
            # Generic tech terms
            'data', 'information', 'code', 'software', 'application', 'app', 'program',
            'website', 'site', 'page', 'email', 'file', 'document', 'report',
            'content', 'text', 'message', 'post', 'comment', 'link', 'url',
            
            # Generic time/frequency terms
            'today', 'yesterday', 'tomorrow', 'always', 'never', 'sometimes', 'often',
            'usually', 'already', 'yet', 'soon', 'late', 'early', 'long', 'short',
            
            # Generic verbs (past tense)
            'said', 'made', 'got', 'went', 'came', 'took', 'gave', 'put', 'set',
            'ran', 'started', 'stopped', 'created', 'built', 'added', 'removed',
            'deleted', 'updated', 'changed', 'moved', 'helped', 'found', 'looked',
            'saw', 'showed', 'told', 'knew', 'thought', 'wanted', 'needed', 'tried',
            
            # Generic modal verbs
            'would', 'could', 'should', 'might', 'must', 'shall', 'will', 'can',
            'may', 'ought', 'dare', 'need', 'used',
            
            # Generic prepositions (extended)
            'about', 'above', 'across', 'after', 'against', 'along', 'among', 'around',
            'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond',
            'during', 'except', 'inside', 'into', 'near', 'outside', 'over', 'through',
            'throughout', 'under', 'until', 'upon', 'within', 'without', 'toward',
            'towards', 'underneath', 'alongside', 'amid', 'amidst',
            
            # Generic internet/social media terms
            'reddit', 'post', 'comment', 'user', 'username', 'link', 'thread', 'subreddit',
            'upvote', 'downvote', 'karma', 'moderator', 'admin', 'bot', 'account',
            
            # Generic question words and phrases
            'anyone', 'someone', 'everyone', 'nobody', 'anybody', 'somebody', 'everybody',
            'anything', 'something', 'everything', 'nothing', 'anywhere', 'somewhere',
            'everywhere', 'nowhere', 'however', 'whatever', 'whenever', 'wherever',
            'whoever', 'whomever', 'whichever',
            
            # Generic exclamations and interjections
            'oh', 'ah', 'eh', 'um', 'hmm', 'yeah', 'yes', 'no', 'ok', 'okay', 'thanks',
            'thank', 'please', 'sorry', 'excuse', 'hello', 'hi', 'hey', 'bye', 'goodbye',
            
            # Generic contractions (expanded)
            'dont', 'doesnt', 'didnt', 'wont', 'wouldnt', 'cant', 'couldnt', 'shouldnt',
            'mustnt', 'neednt', 'isnt', 'arent', 'wasnt', 'werent', 'hasnt', 'havent',
            'hadnt', 'im', 'youre', 'hes', 'shes', 'its', 'were', 'theyre', 'ive',
            'youve', 'weve', 'theyve', 'ill', 'youll', 'hell', 'shell', 'well',
            'theyll', 'id', 'youd', 'hed', 'shed', 'wed', 'theyd'
        ])
        
        # Industry keywords
        self.industry_keywords = {
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
        
    def load_and_process_data(self):
        """Load and process the scraped Reddit data"""
        print("Loading and processing data...")
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            self.scraped_data = json.load(f)
        
        processed_data = []
        for post in self.scraped_data:
            full_text = post['title'] + " " + (post['selftext'] or "")
            comments_text = " ".join(post['comments'])
            full_text_with_comments = full_text + " " + comments_text
            
            # Basic preprocessing
            cleaned_text = self.preprocess_text(full_text_with_comments)
            
            # Classify industries
            industries = self.classify_industry(cleaned_text, self.industry_keywords)
            
            processed_item = {
                'id': post['id'],
                'subreddit': post['subreddit'],
                'title': post['title'],
                'original_text': full_text_with_comments,
                'processed_text': cleaned_text,
                'created_utc': post['created_utc'],
                'industries': list(industries),
                'word_count': len(cleaned_text.split()),
                'sentence_count': len(sent_tokenize(full_text_with_comments)),
                'comment_count': len(post['comments'])
            }
            processed_data.append(processed_item)
        
        self.processed_data = processed_data
        self.df = pd.DataFrame(processed_data)
        
        # Convert timestamps and create datetime columns here
        self.df['datetime'] = pd.to_datetime(self.df['created_utc'])
        self.df['date'] = self.df['datetime'].dt.date
        self.df['hour'] = self.df['datetime'].dt.hour
        self.df['day_of_week'] = self.df['datetime'].dt.day_name()
        
        print(f"Processed {len(processed_data)} posts")
        
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not text:
            return ""
        
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower().strip()
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Lemmatize
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(lemmatized_tokens)
    
    def classify_industry(self, text, industry_keywords):
        """Classify text into industries"""
        found = set()
        for industry, keywords in industry_keywords.items():
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw.lower()) + r'\b', text):
                    found.add(industry)
        return found
    
    def create_word_cloud(self, save_path='wordcloud.png'):
        """Generate word cloud from processed text"""
        print("Creating word cloud...")
        
        # Combine all processed texts
        all_text = ' '.join(self.df['processed_text'].dropna())
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=1200, 
            height=600, 
            background_color='white',
            colormap='viridis',
            max_words=100,
            relative_scaling=0.5,
            random_state=42
        ).generate(all_text)
        
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Common Words in Workflow Automation Discussions', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_industry_distribution(self):
        """Create various plots for industry distribution"""
        print("Creating industry distribution plots...")
        
        # Count industries
        industry_counts = Counter()
        for industries in self.df['industries']:
            for industry in industries:
                industry_counts[industry] += 1
        
        # Create subplot with multiple visualizations
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # 1. Bar chart
        industries, counts = zip(*industry_counts.most_common(10))
        axes[0, 0].bar(industries, counts, color='skyblue', edgecolor='navy', alpha=0.7)
        axes[0, 0].set_title('Top 10 Industries - Bar Chart', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Number of Use Cases')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Horizontal bar chart
        axes[0, 1].barh(industries, counts, color='lightcoral', edgecolor='darkred', alpha=0.7)
        axes[0, 1].set_title('Top 10 Industries - Horizontal Bar Chart', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Number of Use Cases')
        
        # 3. Pie chart
        top_8 = industry_counts.most_common(8)
        labels, sizes = zip(*top_8)
        axes[1, 0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        axes[1, 0].set_title('Top 8 Industries Distribution', fontsize=14, fontweight='bold')
        
        # 4. Donut chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_8)))
        wedges, texts, autotexts = axes[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%', 
                                                  startangle=140, colors=colors, pctdistance=0.85)
        
        # Add circle in center for donut effect
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        axes[1, 1].add_artist(centre_circle)
        axes[1, 1].set_title('Top 8 Industries - Donut Chart', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('industry_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_subreddit_analysis(self):
        """Analyze and visualize subreddit activity"""
        print("Creating subreddit analysis plots...")
        
        subreddit_counts = self.df['subreddit'].value_counts()
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # 1. Subreddit post counts
        axes[0, 0].bar(subreddit_counts.index[:10], subreddit_counts.values[:10], 
                       color='mediumpurple', alpha=0.7)
        axes[0, 0].set_title('Posts by Subreddit', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Number of Posts')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Word count distribution by subreddit
        subreddit_word_counts = self.df.groupby('subreddit')['word_count'].mean().sort_values(ascending=False)
        axes[0, 1].bar(subreddit_word_counts.index[:10], subreddit_word_counts.values[:10], 
                       color='orange', alpha=0.7)
        axes[0, 1].set_title('Average Word Count by Subreddit', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Average Word Count')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Comment count distribution
        subreddit_comment_counts = self.df.groupby('subreddit')['comment_count'].mean().sort_values(ascending=False)
        axes[1, 0].bar(subreddit_comment_counts.index[:10], subreddit_comment_counts.values[:10], 
                       color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Average Comment Count by Subreddit', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Average Comment Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Sentence count distribution
        subreddit_sentence_counts = self.df.groupby('subreddit')['sentence_count'].mean().sort_values(ascending=False)
        axes[1, 1].bar(subreddit_sentence_counts.index[:10], subreddit_sentence_counts.values[:10], 
                       color='salmon', alpha=0.7)
        axes[1, 1].set_title('Average Sentence Count by Subreddit', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Average Sentence Count')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('subreddit_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_text_statistics(self):
        """Create statistical analysis plots"""
        print("Creating text statistics plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        
        # 1. Word count distribution
        axes[0, 0].hist(self.df['word_count'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Word Count Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Word Count')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(self.df['word_count'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {self.df["word_count"].mean():.1f}')
        axes[0, 0].legend()
        
        # 2. Sentence count distribution
        axes[0, 1].hist(self.df['sentence_count'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Sentence Count Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Sentence Count')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(self.df['sentence_count'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {self.df["sentence_count"].mean():.1f}')
        axes[0, 1].legend()
        
        # 3. Comment count distribution
        axes[0, 2].hist(self.df['comment_count'], bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 2].set_title('Comment Count Distribution', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('Comment Count')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].axvline(self.df['comment_count'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {self.df["comment_count"].mean():.1f}')
        axes[0, 2].legend()
        
        # 4. Box plot for word counts by subreddit
        top_subreddits = self.df['subreddit'].value_counts().head(8).index
        filtered_df = self.df[self.df['subreddit'].isin(top_subreddits)]
        
        subreddit_order = filtered_df.groupby('subreddit')['word_count'].median().sort_values(ascending=False).index
        axes[1, 0].boxplot([filtered_df[filtered_df['subreddit'] == sub]['word_count'].values 
                           for sub in subreddit_order], labels=subreddit_order)
        axes[1, 0].set_title('Word Count Distribution by Subreddit', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Word Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Scatter plot: Word count vs Comment count
        axes[1, 1].scatter(self.df['word_count'], self.df['comment_count'], alpha=0.6, color='purple')
        axes[1, 1].set_title('Word Count vs Comment Count', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Word Count')
        axes[1, 1].set_ylabel('Comment Count')
        
        # Add correlation coefficient
        correlation = self.df['word_count'].corr(self.df['comment_count'])
        axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                       transform=axes[1, 1].transAxes, fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 6. Violin plot for sentence counts
        axes[1, 2].violinplot([filtered_df[filtered_df['subreddit'] == sub]['sentence_count'].values 
                              for sub in subreddit_order[:6]], positions=range(1, 7))
        axes[1, 2].set_title('Sentence Count Distribution by Subreddit', fontsize=12, fontweight='bold')
        axes[1, 2].set_ylabel('Sentence Count')
        axes[1, 2].set_xticks(range(1, 7))
        axes[1, 2].set_xticklabels(subreddit_order[:6], rotation=45)
        
        plt.tight_layout()
        plt.savefig('text_statistics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_ngram_analysis(self):
        """Analyze and visualize n-grams"""
        print("Creating n-gram analysis plots...")
        
        # Combine all processed texts
        all_text = ' '.join(self.df['processed_text'].dropna())
        tokens = all_text.split()
        
        # Generate n-grams
        bigrams = list(ngrams(tokens, 2))
        trigrams = list(ngrams(tokens, 3))
        
        # Count n-grams
        bigram_counts = Counter(bigrams)
        trigram_counts = Counter(trigrams)
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # 1. Top bigrams
        top_bigrams = bigram_counts.most_common(15)
        bigram_labels = [' '.join(bg) for bg, _ in top_bigrams]
        bigram_values = [count for _, count in top_bigrams]
        
        axes[0, 0].barh(bigram_labels, bigram_values, color='lightblue', alpha=0.7)
        axes[0, 0].set_title('Top 15 Bigrams', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Frequency')
        
        # 2. Top trigrams
        top_trigrams = trigram_counts.most_common(15)
        trigram_labels = [' '.join(tg) for tg, _ in top_trigrams]
        trigram_values = [count for _, count in top_trigrams]
        
        axes[0, 1].barh(trigram_labels, trigram_values, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Top 15 Trigrams', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Frequency')
        
        # 3. Word frequency distribution
        word_counts = Counter(tokens)
        top_words = word_counts.most_common(20)
        word_labels = [word for word, _ in top_words]
        word_values = [count for _, count in top_words]
        
        axes[1, 0].bar(word_labels, word_values, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Top 20 Words', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Zipf's law visualization
        word_frequencies = sorted(word_counts.values(), reverse=True)
        ranks = range(1, len(word_frequencies) + 1)
        
        axes[1, 1].loglog(ranks[:100], word_frequencies[:100], 'b-', alpha=0.7)
        axes[1, 1].set_title("Zipf's Law - Word Frequency vs Rank", fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Rank (log scale)')
        axes[1, 1].set_ylabel('Frequency (log scale)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ngram_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_temporal_analysis(self):
        """Analyze temporal patterns in the data"""
        print("Creating temporal analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # 1. Posts over time
        daily_posts = self.df.groupby('date').size()
        axes[0, 0].plot(daily_posts.index, daily_posts.values, marker='o', alpha=0.7, color='blue')
        axes[0, 0].set_title('Posts Over Time', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Number of Posts')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Posts by hour of day
        hourly_posts = self.df['hour'].value_counts().sort_index()
        axes[0, 1].bar(hourly_posts.index, hourly_posts.values, color='orange', alpha=0.7)
        axes[0, 1].set_title('Posts by Hour of Day', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Hour')
        axes[0, 1].set_ylabel('Number of Posts')
        
        # 3. Posts by day of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_posts = self.df['day_of_week'].value_counts().reindex(day_order)
        axes[1, 0].bar(weekly_posts.index, weekly_posts.values, color='green', alpha=0.7)
        axes[1, 0].set_title('Posts by Day of Week', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Day of Week')
        axes[1, 0].set_ylabel('Number of Posts')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Heatmap of activity by hour and day
        activity_heatmap = self.df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
        activity_heatmap = activity_heatmap.reindex(day_order)
        
        im = axes[1, 1].imshow(activity_heatmap.values, cmap='YlOrRd', aspect='auto')
        axes[1, 1].set_title('Activity Heatmap (Day vs Hour)', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Hour')
        axes[1, 1].set_ylabel('Day of Week')
        axes[1, 1].set_xticks(range(24))
        axes[1, 1].set_yticks(range(7))
        axes[1, 1].set_yticklabels(day_order)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1, 1], label='Number of Posts')
        
        plt.tight_layout()
        plt.savefig('temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_interactive_plots(self):
        """Create interactive plots using Plotly"""
        print("Creating interactive plots...")
        
        # Industry distribution interactive pie chart
        industry_counts = Counter()
        for industries in self.df['industries']:
            for industry in industries:
                industry_counts[industry] += 1
        
        industries, counts = zip(*industry_counts.most_common(10))
        
        fig = go.Figure(data=[go.Pie(labels=industries, values=counts, hole=0.3)])
        fig.update_layout(title="Interactive Industry Distribution", 
                         font=dict(size=14))
        fig.write_html('interactive_industry_pie.html')
        fig.show()
        
        # Interactive scatter plot
        fig = px.scatter(self.df, x='word_count', y='comment_count', 
                        color='subreddit', hover_data=['title'],
                        title="Interactive Word Count vs Comment Count")
        fig.write_html('interactive_scatter.html')
        fig.show()
        
    def generate_summary_statistics(self):
        """Generate and display summary statistics"""
        print("\n" + "="*60)
        print("REDDIT WORKFLOW AUTOMATION DATA ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\nDataset Overview:")
        print(f"• Total posts analyzed: {len(self.df)}")
        print(f"• Number of subreddits: {self.df['subreddit'].nunique()}")
        print(f"• Date range: {self.df['datetime'].min()} to {self.df['datetime'].max()}")
        
        print(f"\nText Statistics:")
        print(f"• Average word count: {self.df['word_count'].mean():.1f}")
        print(f"• Average sentence count: {self.df['sentence_count'].mean():.1f}")
        print(f"• Average comment count: {self.df['comment_count'].mean():.1f}")
        print(f"• Total words processed: {self.df['word_count'].sum():,}")
        
        print(f"\nTop 5 Subreddits:")
        for subreddit, count in self.df['subreddit'].value_counts().head().items():
            print(f"• {subreddit}: {count} posts")
        
        industry_counts = Counter()
        for industries in self.df['industries']:
            for industry in industries:
                industry_counts[industry] += 1
        
        print(f"\nTop 5 Industries:")
        for industry, count in industry_counts.most_common(5):
            print(f"• {industry}: {count} mentions")
        
        print("\n" + "="*60)
        
    def run_all_visualizations(self):
        """Run all visualization methods"""
        print("Starting comprehensive visualization analysis...")
        
        # Load and process data
        self.load_and_process_data()
        
        # Generate summary statistics
        self.generate_summary_statistics()
        
        # Create all visualizations
        self.create_word_cloud()
        self.plot_industry_distribution()
        self.plot_subreddit_analysis()
        self.plot_text_statistics()
        self.plot_ngram_analysis()
        self.plot_temporal_analysis()
        self.create_interactive_plots()
        
        print("\nAll visualizations completed! Check the generated PNG files and HTML files.")
        print("Files generated:")
        print("• wordcloud.png")
        print("• industry_distribution.png")
        print("• subreddit_analysis.png")
        print("• text_statistics.png")
        print("• ngram_analysis.png")
        print("• temporal_analysis.png")
        print("• interactive_industry_pie.html")
        print("• interactive_scatter.html")

def main():
    """Main function to run the visualization dashboard"""
    visualizer = RedditDataVisualizer('reddit_scraped.json')
    visualizer.run_all_visualizations()

if __name__ == "__main__":
    main()