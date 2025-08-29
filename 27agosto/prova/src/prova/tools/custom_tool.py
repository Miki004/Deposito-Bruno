from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""

    number1 : int =0
    number2 : int =0



class MyCustomTool(BaseTool):
    name: str = "Adder of numbers"
    description: str = (
        "This tool can sum two numbers"
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, number1: int, number2: int) -> int:
        """This function sum two numbers togheter"""
        sum = number1 + number2
        return sum
    


