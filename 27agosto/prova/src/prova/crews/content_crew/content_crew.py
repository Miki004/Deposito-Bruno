# src/guide_creator_flow/crews/content_crew/content_crew.py
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

@CrewBase
class ContentCrew():
    """Content writing crew"""

    agents: List[BaseAgent]
    tasks: List[Task]


    @agent
    def web_searcher(self) -> Agent:
        return Agent(
            config = self.agents_config["web_searcher"],
            verbose = True
        )
    
    @agent
    def web_summarizer(self) -> Agent:
        return Agent(
            config = self.agents_config["web_summarizer"],
            verbose = True
        )
    
    @task
    def web_searcher_task(self) -> Task:
        return Task(
            config = self.tasks_config["web_searcher_task"]
        )
    
    @task
    def summarization_task(self) -> Task:
        return Task(
            config = self.tasks_config["summarization_task"] # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the content writing crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )