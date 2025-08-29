#!/usr/bin/env python
from pydantic import BaseModel
from crewai.flow import Flow, listen, start
from guide_creator_flow.crews.search_crew.search_crew import SearchCrew


class SearchState(BaseModel):
    topic: str = ""
    summary: str = ""


class SearchFlow(Flow[SearchState]):

    @start()
    def get_topic(self):
        #print("Inserisci l'argomento da cercare:")
        self.state.topic = "Matera"#input().strip()  # Legge input utente

    @listen(get_topic)
    def execute_crew(self):
        print(f"Ricerca in corso per: {self.state.topic}")
        result = (
            SearchCrew()
            .crew()
            .kickoff(inputs={"topic": self.state.topic})
        )
        self.state.summary = result.raw  # Output del crew
        print("\n✅ Riassunto dei primi 3 risultati:")
        print(self.state.summary)


def kickoff():
    search_flow = SearchFlow()
    search_flow.kickoff()


if __name__ == "__main__":
    kickoff()
