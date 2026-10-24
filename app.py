import streamlit as st
from src.rag import RAGPipeline
from src.config import TOP_K

st.set_page_config(page_title='Ask ULIP',page_icon='🚚',layout='wide')
st.title('🚚 Ask ULIP')
st.caption('RAG-based assistant over a curated public ULIP / logistics-policy corpus')

@st.cache_resource
def load(): return RAGPipeline()

try:
    rag=load()
except Exception as e:
    st.error(str(e)); st.stop()

with st.sidebar:
    st.header('Retrieval')
    k=st.slider('Top-k chunks',2,8,TOP_K)

examples=['What is ULIP?','How can a port automate vehicle entry using ULIP?','Explain the relationship between ULIP, the National Logistics Policy and PM Gati Shakti.','What are the roles of FASTag and Vahan in logistics?']
q=st.chat_input('Ask a question about ULIP or Indian logistics policy')
if not q:
    st.markdown('### Try a question')
    for e in examples: st.markdown(f'- {e}')
else:
    with st.chat_message('user'): st.write(q)
    with st.chat_message('assistant'):
        with st.spinner('Retrieving, reranking and generating...'):
            answer,results=rag.ask(q,k)
        st.markdown(answer)
        with st.expander('Retrieved sources'):
            for i,r in enumerate(results,1):
                c=r['chunk']
                st.markdown(f'**{i}. {c.title}**')
                st.caption(f'{c.source} · {c.date} · similarity {r["score"]:.3f} · rerank {r["rerank_score"]:.3f}')
                if c.url: st.markdown(f'[Open public source]({c.url})')
                st.write(c.text[:600] + ('...' if len(c.text)>600 else ''))
