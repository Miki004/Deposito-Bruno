from crewai.flow.flow import Flow, listen, start
from litellm import completion

class GeographyFlow(Flow):

    model_name = "gpt-4o"
    @start
    def generate_city(self):
        #rispondi usando self.model_name
        response = completion(
            model = self.model_name,
            messages= {
                "role": "user",
                "content": "Return a random city in the world"
            }

        )
        random_city = response["choices"][0]["message"]["content"]
        return random_city
    

    @listen(generate_city)
    def generate_fun_fact(self, random_city):
        print(f"Finding a fun fact about: {random_city}")
        response = completion(
            model = self.model_name,
            messages= {
                "role": "user",
                "content": f"Return a fun fact about {random_city}"
            }

        )
        fun_fact = response["choices"][0]["message"]["content"]
        return fun_fact 
 


flow = GeographyFlow()
result = flow.kickoff()
print(f"{result}")