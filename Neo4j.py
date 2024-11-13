from neo4j import GraphDatabase
import pandas as pd

class Neo4jDatabaseHandler:
    def __init__(self, uri, user, password):
        # Set up the connection to the Neo4j database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        # Close the database connection
        if self.driver:
            self.driver.close()

    def add_paper(self, paper_id, title, abstract):
        """
        Adds a research paper to the Neo4j database.
        """
        with self.driver.session() as session:
            session.write_transaction(self._create_paper_node, paper_id, title, abstract)

    @staticmethod
    def _create_paper_node(tx, paper_id, title, abstract):
        # Add a new node for a research paper
        query = (
            "CREATE (p:Paper {paper_id: $paper_id, title: $title, abstract: $abstract})"
        )
        tx.run(query, paper_id=paper_id, title=title, abstract=abstract)

    def query_papers(self, topic):
        """
        Queries papers from the database that match the given topic.
        """
        with self.driver.session() as session:
            result = session.read_transaction(self._find_papers_by_topic, topic)
            return result

    @staticmethod
    def _find_papers_by_topic(tx, topic):
        # Find papers related to a specific topic
        query = (
            "MATCH (p:Paper) "
            "WHERE p.title CONTAINS $topic OR p.abstract CONTAINS $topic "
            "RETURN p.title AS title, p.abstract AS abstract"
        )
        result = tx.run(query, topic=topic)
        return [{"title": record["title"], "abstract": record["abstract"]} for record in result]