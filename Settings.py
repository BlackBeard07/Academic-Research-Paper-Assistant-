import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Neo4j Database Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+ssc://f4eda1f4.databases.neo4j.io")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "qfDuBBza53oO_wtuHOmwdKHIb9u9JZMyBsqkQiqPQyM")

# API Configuration
ARXIV_API_URL = "https://export.arxiv.org/api/query"
MAX_RESULTS = int(os.getenv("MAX_RESULTS", 5))

# Transformer Model Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "t5-small")
MAX_SUMMARY_LENGTH = int(os.getenv("MAX_SUMMARY_LENGTH", 300))
MAX_FUTURE_RESEARCH_LENGTH = int(os.getenv("MAX_FUTURE_RESEARCH_LENGTH", 300))