from crewai import Agent, Crew, Process, Task


class SearchCrew:
    """Crew con un agente che riassume - senza tool esterni"""

    def __init__(self):
        # Crea l'agente direttamente (senza tool esterni)
        self.researcher_agent = Agent(
            role="Knowledge Expert",
            goal="Provide detailed information and summaries on given topics using available knowledge",
            backstory="You are an expert researcher with vast knowledge who can provide comprehensive information and analysis on any topic.",
            verbose=True
        )
        
        # Crea il task
        self.research_task = Task(
            description="Fornisci informazioni dettagliate e un riassunto completo sull'argomento: {topic}. Include almeno 3 punti chiave importanti, esempi pratici se possibile, e una conclusione informativa.",
            expected_output="Un riassunto dettagliato e ben strutturato con almeno 3 punti chiave, esempi e una conclusione sull'argomento richiesto.",
            agent=self.researcher_agent
        )

    def crew(self) -> Crew:
        """Ritorna il crew configurato"""
        return Crew(
            agents=[self.researcher_agent],
            tasks=[self.research_task],
            process=Process.sequential,
            verbose=True
        )