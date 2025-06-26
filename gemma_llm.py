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
        formatted_prompt = f"""You are a financial analyst working with specialized reports (ARGUS, CRU, etc.).

        Context from reports:
        {context}

        Question: {prompt}

        Guidelines for response:
        1. Be precise with numbers and metrics
        2. Cite sources when possible
        3. Highlight trends or anomalies
        4. Maintain professional tone

        Analysis:"""

        response = self.llm.invoke(formatted_prompt)
        return response.content