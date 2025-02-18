import praw
import pandas as pd
from typing import Optional
import logging

class SocialMediaScraper:
    def __init__(self, client_id: str, client_secret: str, user_agent: str, subreddit: str = "stocks"):
        """
        Initialize Reddit scraper with authentication credentials.
        
        Args:
            client_id (str): Reddit API client ID
            client_secret (str): Reddit API client secret
            user_agent (str): Unique user agent string
            subreddit (str, optional): Subreddit to scrape. Defaults to "stocks".
        """
        try:
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            self.subreddit = subreddit
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        except Exception as e:
            logging.error(f"Error initializing Reddit client: {e}")
            raise

    def scrape(self, limit: int = 100, sort_by: str = 'new') -> pd.DataFrame:
        """
        Scrape posts from specified subreddit.
        
        Args:
            limit (int, optional): Number of posts to scrape. Defaults to 100.
            sort_by (str, optional): How to sort posts. Defaults to 'new'.
        
        Returns:
            pd.DataFrame: Scraped posts with relevant information
        """
        try:
            subreddit = self.reddit.subreddit(self.subreddit)
            
            # Dynamic sorting based on input
            sorter = {
                'new': subreddit.new,
                'hot': subreddit.hot,
                'top': subreddit.top
            }.get(sort_by, subreddit.new)
            
            posts = []
            for post in sorter(limit=limit):
                # Enhanced error handling with attribute checks
                posts.append({
                    "title": getattr(post, 'title', ''),
                    "selftext": getattr(post, 'selftext', ''),
                    "score": getattr(post, 'score', 0),
                    "comments": getattr(post, 'num_comments', 0),
                    "created_utc": getattr(post, 'created_utc', 0),
                    "url": getattr(post, 'url', ''),
                    "author": str(getattr(post, 'author', 'deleted')),
                    "permalink": getattr(post, 'permalink', '')
                })
            
            df = pd.DataFrame(posts)
            self.logger.info(f"Successfully scraped {len(df)} posts from r/{self.subreddit}")
            return df
        
        except Exception as e:
            self.logger.error(f"Error during scraping: {e}")
            return pd.DataFrame()

    def get_comments(self, post_url: str, comment_limit: int = 10) -> list:
        """
        Retrieve comments for a specific post.
        
        Args:
            post_url (str): URL of the Reddit post
            comment_limit (int, optional): Number of comments to retrieve. Defaults to 10.
        
        Returns:
            list: List of comment dictionaries
        """
        try:
            submission = self.reddit.submission(url=post_url)
            submission.comments.replace_more(limit=0)  # Flatten comment tree
            
            comments = []
            for comment in submission.comments.list()[:comment_limit]:
                comments.append({
                    'text': comment.body,
                    'score': comment.score,
                    'author': str(comment.author)
                })
            
            return comments
        
        except Exception as e:
            self.logger.error(f"Error retrieving comments: {e}")
            return []