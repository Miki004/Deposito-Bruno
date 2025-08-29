#!/usr/bin/env python
import json
import os
from typing import List, Dict
from pydantic import BaseModel, Field
from crewai import LLM
from crewai.flow.flow import Flow, listen, start, router
from src.rag_flow.crews.content_crew.content_crew import ContentCrew
from src.rag_flow.crews.rag_crew.rag_crew import RagCrew


# Define our flow state
class GuideCreatorState(BaseModel):
    """Represents the state of the guide creation flow."""
    topic: str = ""
    choice : str = ""
    output : str = ""

class GuideCreatorFlow(Flow[GuideCreatorState]):
    """Flow for creating a comprehensive guide on any topic"""

    @start()
    def get_user_input(self):
        """Get input from the user about the guide topic and audience"""
        print("\n=== RAG AGENT  ===\n")
        # Get user input
        self.state.topic = "dimmi qualcosa sulle malattie cardiovascolari "

        return self.state

    @listen(get_user_input)
    def executing_rag(self,state):
        """Executes the RAG (Retrieval Augmented Generation) process based on the user's topic."""
        query = state.topic
        crew = RagCrew() 
        crew_output = crew.crew().kickoff(
            inputs = {
                "query" : query
            }
        )
        return crew_output
    
    @router(executing_rag)
    def routing(self,crew_output):
        """Routes the flow based on the output of the RAG crew."""
        self.state.output = crew_output
        if crew_output == "failed":
            return "search"
        else:
            return "print"
        
    # @router(get_user_input)
    # def routing(self, state):
    #     if state.topic.lower() == "SEARCH":
    #         return "search"
    #     else:
    #         return ""

    # @listen("add")
    # def run_adder_crew(self):
    #     """Run the adder crew"""
    #     print("Runnig the adding crew")
    #     number1 = input("Write the first number: ")
    #     number2 = input("Write the second numer:")
    #     crew = SumCrew()
    #     crew_output = crew.crew().kickoff(
    #         inputs = {
    #             "number1" : number1,
    #             "number2" : number2
    #         }
    #     )
    #     print(f"Output adder: {crew_output}")
    
    @listen("print")
    def print(self):
        """Prints the output from the RAG agent."""
        print(f"OUTPUT OF THE RAG AGENT: {self.state.output}")

    @listen("search")
    def run_crew(self):
        """Run the web search crew to gather information and create the guide"""
        print("Running the web search crew...")

        # Initialize and run the crew
        crew = ContentCrew()
        crew_output = crew.crew().kickoff(
            inputs={
                "topic": self.state.topic,
                "web_search_task": {
                    "topic": self.state.topic
                },
                "summarization_task": {
                    "topic": self.state.topic
                }
            }
        )        
        print("Web search crew completed.")

def kickoff():
    """Run the guide creator flow"""
    GuideCreatorFlow().kickoff()
    print("\n=== Flow Complete ===")

def plot():
    """Generate a visualization of the flow"""
    flow = GuideCreatorFlow()
    flow.plot("guide_creator_flow")
    print("Flow visualization saved to guide_creator_flow.html")

if __name__ == "__main__":
    kickoff()