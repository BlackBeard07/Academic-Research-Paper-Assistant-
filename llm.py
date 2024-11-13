import requests
from transformers import pipeline
from Settings import MODEL_NAME
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from Settings import MAX_SUMMARY_LENGTH, MAX_FUTURE_RESEARCH_LENGTH


# LLM Research Assistant using Hugging Face
class LLMResearchAssistant:
    def __init__(self, model_name=MODEL_NAME):
        # Initialize the model and tokenizer from Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.pipeline = pipeline('text2text-generation', model=self.model, tokenizer=self.tokenizer)

    def generate_summary(self, text, max_length=MAX_SUMMARY_LENGTH):
        """
        Generates a summary for the given text using the transformer model.
        """
        prompt = f"summarize: {text}"
        result = self.pipeline(prompt, max_length=max_length, num_return_sequences=1)
        return result[0]['generated_text']

    def question_answer(self, context, question):
        """
        Provides an answer to the user's question based on the given context.
        """
        prompt = f"question: {question} context: {context}"
        result = self.pipeline(prompt, max_length=MAX_SUMMARY_LENGTH, num_return_sequences=1)
        return result[0]['generated_text']

    def propose_future_research(self, papers_list):
        """
        Generates suggestions for future research directions based on reviewed papers.
        """
        combined_text = " ".join([paper['title'] + ' ' + paper.get('abstract', '') for paper in papers_list])
        prompt = f"Based on the following paper summaries, propose potential future research opportunities: {combined_text}"
        result = self.pipeline(prompt, max_length=MAX_FUTURE_RESEARCH_LENGTH, num_return_sequences=1)
        return result[0]['generated_text']