from langchain.chat_models import ChatOpenAI


class Gemma3Llm:
    def __init__(self, api_key: str, base_url: str):
        self.llm = ChatOpenAI(
            model='Gemma-3-27b',
            openai_api_key=api_key,
            openai_api_base=base_url,
            max_tokens=6000,
            temperature=0.1,  # Added default temperature
            streaming=True
        )

    def generate_response(self, prompt: str, context: str) -> str:
        # Enhanced prompt template for financial reports
        formatted_prompt = f"""
        You are a senior financial analyst with deep expertise in fertilizers, 
        raw materials (including phosphate rock), and crop markets, 
        frequently working with intelligence reports from ARGUS, CRU, and similar industry sources.

        Below is the extracted context from one or more proprietary reports:
        {context}

        Your task is to answer the following question with a structured, data-driven analysis:
        Question: {prompt}

        When formulating your response, follow these guidelines:
        1. Be specific and accurate with quantitative figures (e.g., prices, volumes, growth rates).
        2. Reference report names (e.g., ARGUS, CRU) and dates where applicable to support claims.
        3. Emphasize significant trends, shifts, or anomalies in the data.
        4. Use a clear, professional, and analytical tone suitable for executive-level reporting.
        5. Structure the analysis in paragraphs or bullet points if needed for clarity.

        Begin your analysis below:
        """

        response = self.llm.invoke(formatted_prompt)
        return response.content