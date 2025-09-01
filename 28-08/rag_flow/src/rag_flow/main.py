#!/usr/bin/env python
import json
import os
from typing import List, Dict
from pydantic import BaseModel, Field
from crewai import LLM
from crewai.flow.flow import Flow, listen, start, router
from src.rag_flow.crews.content_crew.content_crew import ContentCrew
from src.rag_flow.crews.drug_crew.drug_crew import DrugCrew
from src.rag_flow.crews.rag_crew.rag_crew import RagCrew


# Define our flow state
class GuideCreatorState(BaseModel):
    """Represents the state of the guide creation flow."""
    topic: str = ""
    output : str = ""
    drug_name: str = ""
    description: str = ""

class GuideCreatorFlow(Flow[GuideCreatorState]):
    """Flow for creating a comprehensive guide on any topic"""

    @start()
    def get_user_input(self):
        """Get input from the user about the guide topic and audience"""
        print("\n=== Virtual Pharmacy ===\n")
        print("Dimmi di cosa hai bisogno?")
        print("1.Ricerca su RAG")
        print("2.Richiesta calcolo del dosaggio di un farmaco")
        choice = int(input("Inserire scelta: "))

        if choice == 2:
            self.state.description = input("Forniscimi una descrizione del paziente (peso, eta', sesso, condizioni mediche): ")
            self.state.drug_name = input("Inserisci il nome del farmaco: ")
        elif choice == 1:
            self.state.topic = input("Inserisci la domanda di carattere medico: ")

        return choice
    
    @router(get_user_input)
    def routing(self,choice):
        if choice == 1:
            return "rag_search"
        else:
            return "drug_calculation"

    @listen("rag_search")
    def executing_rag(self,state):
        """Executes the RAG (Retrieval Augmented Generation) process based on the user's topic."""
        query = self.state.topic
        crew = RagCrew() 
        crew_output = crew.crew().kickoff(
            inputs = {
                "query" : query
            }
        )
        return crew_output.raw

    @listen("executing_rag")
    def print_rag_output(self,crew_output):
        """Prints the output from the RAG agent."""
        print(f"OUTPUT OF THE RAG AGENT: {crew_output}")

    @listen("drug_calculation")
    def executing_drug_calculation(self):
        crew = DrugCrew()
        crew_output = crew.crew().kickoff(
            inputs = {
                "description" : self.state.description,
                "drug_name" : self.state.drug_name
            }
        )
        return crew_output.raw

    @listen("executing_drug_calculation")
    def print_drug_calculation_output(self, crew_output):
        """Prints the output from the drug calculation agent."""
        print(f"OUTPUT OF THE DRUG CALCULATION: {crew_output}")

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