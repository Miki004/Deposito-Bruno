from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import tool
from crewai.agents.agent_builder.base_agent import BaseAgent
from src.prova.tools.custom_tool import MyCustomTool
from typing import List
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class SumCrew():
    """SumCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]


    @agent
    def adder(self) -> Agent:
        return Agent(
            config=self.agents_config["adder"], # type: ignore[index]
            tools= [MyCustomTool()],
            verbose=True
        )
    @task
    def adding_task(self) -> Agent:
        return Task(
            config = self.tasks_config["adding_task"]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the SumCrew crew"""
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
