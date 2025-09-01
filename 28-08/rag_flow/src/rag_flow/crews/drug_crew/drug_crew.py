from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from src.rag_flow.tools.custom_tool import duck_duck_search

duck_duck_tool = duck_duck_search
@CrewBase
class DrugCrew():
    """DrugCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def drug_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['drug_researcher'], # type: ignore[index]
            tools=[duck_duck_tool],
            verbose=True    
        )
    
    @agent
    def drug_calculator(self) -> Agent:
        return Agent(
            config=self.agents_config['drug_calculator'], # type: ignore[index]
            verbose=True
        )


    
    @task
    def drug_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['drug_research_task'], # type: ignore[index]
        )
    @task
    def drug_calculation_task(self) -> Task:
        return Task(
            config=self.tasks_config['drug_calculation_task'], # type: ignore[index]
        )


    @crew
    def crew(self) -> Crew:
        """Creates the DrugCrew crew"""
        return Crew(
            agents=self.agents, 
            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True,
        )
