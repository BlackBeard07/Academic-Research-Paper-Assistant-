import requests
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from Settings import MAX_RESULTS

class ArxivPaperFetcher:
    def __init__(self, base_url="http://export.arxiv.org/api/query", max_results=MAX_RESULTS):
        # Setting up the base API URL and max number of results
        self.base_url = base_url
        self.max_results = max_results

    def fetch_papers(self, topic):
        """
        Fetches research papers from the Arxiv API based on a given topic.
        """
        # Define the date range (last 5 years)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5 * 365)  # Approximate 5 years

        # Constructing the request URL
        url = f"{self.base_url}?search_query=all:{topic}&start=0&max_results={self.max_results}"
        response = requests.get(url)
        if response.status_code == 200:
            return self.parse_arxiv_response(response.text)
        else:
            print("Error fetching papers:", response.status_code)
            return []

    def parse_arxiv_response(self, xml_response):
        """
        Parses the XML response from the Arxiv API.
        """
        papers = []
        root = ET.fromstring(xml_response)

        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            title = entry.find("{http://www.w3.org/2005/Atom}title").text
            authors = [author.find("{http://www.w3.org/2005/Atom}name").text for author in entry.findall("{http://www.w3.org/2005/Atom}author")]
            abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text
            published_date = entry.find("{http://www.w3.org/2005/Atom}published").text
            url = entry.find("{http://www.w3.org/2005/Atom}id").text

            # Filter papers published in the last 5 years
            if self.is_within_last_five_years(published_date):
                papers.append({
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "published_date": published_date,
                    "url": url,
                })

        return papers

    def is_within_last_five_years(self, published_date):
        """
        Checks if the paper was published within the last 5 years.
        """
        pub_year = datetime.strptime(published_date, "%Y-%m-%dT%H:%M:%SZ").year
        return pub_year >= (datetime.now().year - 5)

    def answer_question(self, papers, question):
        """
        Provides answers for questions about the paper and shows the exact part of the paper where the information was retrieved.
        """
        from transformers import pipeline
        qa_pipeline = pipeline("question-answering")
        answers = []

        for paper in papers:
            context = paper['abstract']
            result = qa_pipeline(question=question, context=context)
            answer_text = result['answer']
            answer_start = context.find(answer_text)
            answer_end = answer_start + len(answer_text)

            highlighted_context = (
                context[:answer_start] + "*" + answer_text + "*" + context[answer_end:]
                if answer_start != -1 else context
            )

            answers.append({
                "title": paper['title'],
                "answer": answer_text,
                "highlighted_context": highlighted_context
            })

        return answers
    

if __name__ == "__main__":
    paper_fetcher = ArxivPaperFetcher()
    topic = "Machine Learning"
    papers = paper_fetcher.fetch_papers(topic)
    if papers:
        question = "What is the purpose of the study?"
        answers = paper_fetcher.answer_question(papers, question)
        for answer in answers:
            print(f"Title: {answer['title']}")
            print(f"Answer: {answer['answer']}")
            print(f"Highlighted Context: {answer['highlighted_context']}")
    else:
        print("No papers found.")    
