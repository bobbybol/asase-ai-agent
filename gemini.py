from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client()

# Tool definitions
weather_tool = types.Tool(
   function_declarations=[
       types.FunctionDeclaration(
            name="get_weather",
            description="Get current temperature for a given city.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "city": types.Schema(
                        type="STRING",
                        description="The city to get the weather for."
                    )
                },
                # Notice "city" must match the declared property name above
                required=["city"] 
            )
        )
   ]
)
tools = [weather_tool]

def get_weather_stub(city: str):
    cities = {
        "Grottaferrata": {
            "temperature": 32,
            "humidity": 76,
            "wind_speed": 4,
            "clouds": 0
        },
        "Wilnis": {
            "temperature": 23,
            "humidity": 45,
            "wind_speed": 8,
            "clouds": 50
        },
    }

    if city not in cities:
        return f"Sorry, I don't know the weather in {city}."

    return cities[city]

def generate_llm_response(history: list):
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=history,
        config=types.GenerateContentConfig(tools=tools)
    )
    return response

def main():
    history = []

    while True:
        user_input = input("How can I help you today? ")

        if user_input == "exit":
            return

        history.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_input)]
        ))

        response = generate_llm_response(history)

        # 1. Safely check the first candidate response chunk
        parts = response.candidates[0].content.parts
        
        # 2. Check if the model wants to execute a function
        if parts and parts[0].function_call:
            call = parts[0].function_call

            tool_name = call.name
            tool_args = call.args
            print(f"Model wants to call function: {tool_name} with args {tool_args}")
            
            if(tool_name == "get_weather"):
                city = tool_args["city"]
                weather = get_weather_stub(city)
                print(weather)
                history.append(types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=f"You wanted to use the tool {tool_name} with the argument {city}.")]
                ))
                history.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=f"The original questions was: {user_input}. You wanted to use a tool. The result of the tool use is: {weather}")]
                ))

                response = generate_llm_response(history)
                # Yes we need to make this way smarter and recusive, but for this test we just assume we'll get text
                history.append(types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=response.text)]
                ))
                print(response.text)
            else:
                history.append(types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=f"You wanted to use the tool {tool_name} with the arguments {tool_args}.")]
                ))
                history.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=f"The original questions was: {user_input}. You wanted to use a tool. This tool doesn't exist.")]
                ))

                response = generate_llm_response(history)
                history.append(types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=response.text)]
                ))
                print(response.text)

        # 3. If no function call, safely return the normal text output
        else:
            history.append(types.Content(
                role="model",
                parts=[types.Part.from_text(text=response.text)]
            ))
            print(response.text)  

if __name__ == "__main__":
    main();