from fastapi import FastAPI
from pydantic import BaseModel
import os
import openai
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

app = FastAPI()
load_dotenv()

class Question(BaseModel):
    input: str

@app.post("/question")
def root(question: Question):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    url = 'https://sports.news.naver.com/kbaseball/news/index?isphoto=N'
    loader = WebBaseLoader(url)
    docs = loader.load()
    baseballNews = docs[0].page_content

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    chat_prompt = ChatPromptTemplate.from_messages({
        ("system", "이 시스템은 야구와 관련한 질문에 답변할 수 있습니다."),
        ("user", question.input)
    })
    messages = chat_prompt.format_messages(user_input = "김도영의 타할을 알려줘")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    chain = chat_prompt | llm | StrOutputParser()
    # answer = chain.invoke({
    #     "user_input": question.input
    # })

    pc = Pinecone(
        api_key = os.getenv("PINECONE_API_KEY")
    )
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = splits[0].page_content[600:1600]
    embeddings = model.encode([sentences])

    index_name = 'kbo-news'
    pc.delete_index(name=index_name)
    pc.create_index(
        name=index_name, 
        dimension=embeddings.shape[1],
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    index = pc.Index(index_name)
    
    ids = [f"id-{i}" for i in range(len(splits[0].page_content))]
    index.upsert(vectors=list(zip(ids, embeddings.tolist())))
    
    return question