from typing import Type
from ddgs import DDGS
from crewai.tools import BaseTool, tool 
from pydantic import BaseModel, Field
from src.rag_flow.tools.Rag import  Rag, Settings


@tool("DuckDuckGo Search")
def duck_duck_search(query: str) -> str:
    """Searches DuckDuckGo for the given query and returns the results."""
    with DDGS(verify=False) as ddgs:
        results = ddgs._search(query=query, 
                               category='text',
                               region='it', 
                               max_results=3, 
                               safesearch='on'
                               )
        return results
    
class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""

    query: str = " "

    
class MyRAGTool(BaseTool):
    name: str = "Tool Rag"
    description: str = (
        "Questo tool è un RAG specializzato in documenti medici"
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, query: str) -> str:
        """Executes the custom tool's logic with the provided argument."""
        settings = Settings()
        rag = Rag(settings)
        result = rag.rag_answer(query)
        return result

