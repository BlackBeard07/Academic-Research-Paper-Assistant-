import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from Neo4j import Neo4jDatabaseHandler
from Arxiv import ArxivPaperFetcher
from Settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, MODEL_NAME, MAX_SUMMARY_LENGTH, MAX_FUTURE_RESEARCH_LENGTH

from llm import LLMResearchAssistant

# Main Streamlit application interface
def main():
    st.title("Academic Research Paper Assistant")
    st.write("Helping you find, summarize, and explore research papers efficiently!")

    # Creating instances of the assistant and database handler
    research_assistant = LLMResearchAssistant(model_name=MODEL_NAME)
    db_handler = Neo4jDatabaseHandler(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
    paper_fetcher = ArxivPaperFetcher()

    # User input section
    query_topic = st.text_input("Enter your research topic:")
    if query_topic:
        papers = paper_fetcher.fetch_papers(query_topic)
        if papers:
            for idx, paper in enumerate(papers):
                title = paper.get('title', 'N/A')
                abstract = paper.get('abstract', 'N/A')
                url = paper.get('url', '#')  
                
                # Display the title as a clickable link if a URL is available
                st.markdown(f"*{idx + 1}. [{title}]({url})*")
                
                # Store paper details in the Neo4j database
                db_handler.add_paper(str(idx + 1), title, abstract)
        else:
            st.write("No papers found for this topic.")

    # Q&A Feature
    question = st.text_input("Ask a question about a specific paper (use the paper number):")
    if question and query_topic:
        # Retrieving papers from the database and allowing the user to specify the paper number
        paper_number = st.number_input("Enter the paper number you want to ask about:", min_value=1, step=1)
        papers_data = db_handler.query_papers(query_topic)
        if papers_data and 1 <= paper_number <= len(papers_data):
            selected_paper = papers_data[paper_number - 1]
            context = selected_paper['abstract']
            qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
            result = qa_pipeline(question=question, context=context, top_k=1, max_answer_len=100)
            answer_text = result['answer']
            answer_start = context.find(answer_text)
            answer_end = answer_start + len(answer_text)

            highlighted_context = (
                context[:answer_start] + "**" + answer_text + "**" + context[answer_end:]
                if answer_start != -1 else context
            )

            #st.write(f"*Title:* {selected_paper['title']}")
            #st.write(f"*Answer:* {answer_text}")
            st.write(f"*Answer:* {highlighted_context}")
        else:
            st.write("Invalid paper number or no relevant papers found to answer the question.")

    # Suggest Future Research Directions
    if st.button("Suggest Future Research Opportunities") and query_topic:
        papers_data = db_handler.query_papers(query_topic)
        future_research = research_assistant.propose_future_research(papers_data)
        st.write(f"*Future Research Directions:* {future_research}")

    # Close the Neo4j connection
    db_handler.close()

if __name__ == "__main__":
    main()
