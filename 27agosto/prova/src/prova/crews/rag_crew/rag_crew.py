from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import RagTool


# Initialize the tool and add content
rag_tool = RagTool()

rag_tool.add(data_type="file", path="27agosto\\prova\\output\\Piano di Studi CdL in Computer Science a.a. 2025-2026.pdf")

@CrewBase
class RagCrew():
    """RagCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Define an agent with the RagTool
    @agent
    def rag_agent(self) -> Agent:
        '''
        This agent uses the RagTool to answer questions about the knowledge base.
        '''
        
        return Agent(
            config=self.agents_config["rag_agent"],
            allow_delegation=False,
            tools=[rag_tool]
        )

    @task
    def rag_task(self) -> Task:
        return Task(
            config=self.tasks_config['rag_task'], # type: ignore[index]
        )


    @crew
    def crew(self) -> Crew:
        """Creates the RagCrew crew"""

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
