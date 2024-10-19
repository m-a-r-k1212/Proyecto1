import os

from dotenv import load_dotenv
load_dotenv()

from consts import INDEX_NAME
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from firecrawl import FirecrawlApp
from langchain.schema import Document

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_docs2()-> None:
    app = FirecrawlApp(api_key=os.environ['FIRECRAWL_API_KEY'])
    url = "https://concepto.de/basquetbol/, https://www.gigantes.com/nba/espn-nba-10/, https://spain.id.nba.com/noticias/ranking-anotadores-nba, "

    page_content = app.scrape_url(url=url,
                                  params={
                                      "onlyMainContent": True
                                  })
    print(page_content)
    doc = Document(page_content=str(page_content), metadata={"source": url})

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = text_splitter.split_documents([doc])

    PineconeVectorStore.from_documents(
        docs, embeddings, index_name="bas-doc-index"
    )


if __name__ == "__main__":
    ingest_docs2()