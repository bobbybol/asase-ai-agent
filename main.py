import json
import requests

tools = [
    {
        "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The city to get the weather for, e.g. Paris"
                            }
                        },
                        "required": ["city"]
                    }
                }
    },
    {
        "type": "function",
                "function": {
                    "name": "convert_temperature",
                    "description": "Convert a temperature from one unit to another, supports Fahrenheit and Celsius.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "temperature": {"type": "number", "description": "The temperature to convert."},
                            "unit": {"type": "string", "description": "The unit of the temperature to convert from."},
                            "target_unit": {"type": "string", "description": "The unit of the temperature to convert to."}
                        },
                        "required": ["temperature", "unit", "target_unit"]
                    }
                }
    }
]


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


def convert_temperature(temperature: float, unit: str, target_unit: str):
    print(f"Converting {temperature} {unit} to {target_unit}")
    if unit == "C" and target_unit == "F":
        return (temperature * 9/5) + 32
    elif unit == "F" and target_unit == "C":
        return (temperature - 32) * 5/9
    else:
        return temperature


def generate_response(user_prompt: str, history: list):
    # Manually instruct Gemma3 to use tools
    history.append({
        "role": "user",
        "content": f"""
          You have access to functions. If you decide to invoke any of the function(s),
          you MUST put it in the format of
          {{"name": function name, "parameters": dictionary of argument name and its value}}

          You SHOULD NOT include any other text in the response if you call a function
          {json.dumps(tools)}
          With that in mind, revisit the chat history and request a tool call if needed, 
          provide a response to the user's original question otherwise.
          If you want to use a tool make sure to describe the use of the tool using EXACTLY the format shown above.
          As a reminder, here it is: {{"name": function name, "parameters": dictionary of argument name and its value}}
          Don't return any extra text, don't return an object in any different shape.
        """
    })
    response = requests.post('http://localhost:11434/api/chat', json={
        "model": "gemma3:12b-it-qat",
        "messages": history,
        "stream": False,
        # "tools": tools
    })

    output = response.json()

    message = output["message"]
    content = message["content"]

    try:
        tool_call = json.loads(content)
        if isinstance(tool_call, list) and len(tool_call) > 0:
            tool_call = tool_call[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["parameters"]

        print(f"Tool call: {tool_name} with arguments: {tool_args}")

        tool_result = "No result, invalid tool."
        if tool_name == "get_weather":
            city = tool_args["city"]
            tool_result = get_weather_stub(city)
        elif tool_name == "convert_temperature":
            temperature = tool_args["temperature"]
            unit = tool_args["unit"]
            target_unit = tool_args["target_unit"]
            tool_result = convert_temperature(temperature, unit, target_unit)
        
        history.append(
            {"role": "assistant", "content": f"I want to call the tool {tool_name} with the arguments {tool_args}."})
        history.append({
            "role": "user",
            "content": f"""
              My original prompt was: {user_prompt}.
              You wanted to use the tool {tool_name} with the arguments {tool_args}.
              The result of the tool call is: {tool_result}.
            """
        })

        return generate_response(user_prompt, history)
    except Exception as e:
        # Bit ugly but if we don't find tool calls we get the exception and just return the (final) text response 😇
        return content;


def main():
    history = []

    while True:
        user_input = input("Your prompt: ")

        if user_input == "exit":
            return

        history.append({"role": "user", "content": user_input})

        output = generate_response(user_input, history)

        print(output)


if __name__ == "__main__":
    main()