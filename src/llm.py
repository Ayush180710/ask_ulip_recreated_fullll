import os
from .config import GEMINI_API_KEY, GEMINI_MODEL

SYSTEM = '''You are Ask ULIP, a grounded logistics knowledge assistant.
Use only the supplied retrieved context. If the context is insufficient, say that the answer is not available in the indexed public corpus.
Do not invent facts. Keep answers concise. Cite factual statements as [SOURCE n].'''

class Generator:
    def __init__(self):
        self.client=None
        self.fallback_reason='No Gemini API key is configured.'
        if GEMINI_API_KEY:
            try:
                from google import genai
                self.client=genai.Client(api_key=GEMINI_API_KEY)
            except Exception:
                self.client=None
                self.fallback_reason='The Gemini client could not be initialized.'

    def answer(self,question,results):
        if not results:
            return 'I could not find relevant information in the indexed public ULIP corpus.'
        if self.client:
            context='\n\n'.join(
                f'[SOURCE {i}]\nTitle: {r["chunk"].title}\nSource: {r["chunk"].source}\nURL: {r["chunk"].url}\nContent: {r["chunk"].text}'
                for i,r in enumerate(results,1)
            )
            prompt=f'{SYSTEM}\n\nQUESTION:\n{question}\n\nCONTEXT:\n{context}'
            try:
                response=self.client.models.generate_content(model=GEMINI_MODEL,contents=prompt)
                if response.text: return response.text
            except Exception:
                self.fallback_reason='Gemini is temporarily unavailable, so the local evidence extract is shown instead.'
        # Offline fallback makes the demo usable without an API key. It does not
        # generate new prose: it shows the leading retrieved evidence verbatim.
        best=results[0]['chunk']
        excerpt = best.text if len(best.text) <= 900 else best.text[:900].rsplit(' ', 1)[0] + '…'
        return f"**Offline evidence extract:**\n\n_{self.fallback_reason}_\n\n{excerpt} [SOURCE 1]"
