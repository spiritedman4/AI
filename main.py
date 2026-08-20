from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv
import os
from langchain.tools import tool
from langchain.messages import AnyMessage,SystemMessage, ToolMessage, HumanMessage
from typing import Annotated, TypedDict , Literal
import operator
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()

openrouter_key = os.environ.get('openrouter_api_key')

class MessageState(TypedDict):
    messages : Annotated[list[AnyMessage] , operator.add]
    llm_calls: int

@tool
def multiply(a: int, b: int) -> int:
    '''
        multiply two values

        Args:
            a: int
            b: int
    '''

    return a*b

@tool
def add(a: int, b: int)-> int:
    '''
        add two values

        Args
            a: int
            b: int
    '''

    return a+b

@tool
def sub(a: int, b: int)-> int:
    '''
        sub two values

        Args:
            a: int
            b: int
    '''
    return a-b

tools = [multiply, add, sub]
tools_by_name = {tool.name : tool for tool in tools}


class ModelFactory:
    @classmethod
    def get_model(cls, integration):
        if integration == 'openai':
            return  ChatOpenAI(
                model= 'openai/gpt-4o-mini' , 
                api_key=openrouter_key,
                base_url="https://openrouter.ai/api/v1"
            )
        elif integration == 'openrouter' :
            return ChatOpenRouter(
                model= 'openai/gpt-4o-mini' , 
                api_key=openrouter_key
            )

        
        model_with_tools = model.bind_tools(tools=tools)

        return model_with_tools




model = ModelFactory.get_model('openai')

def llm_call(state: MessageState):
    '''
        Decides whether to call a llm or not
    '''

    return {
            'messages' : [
                model.invoke(
                    [
                        SystemMessage(content='You are a helpful assistant who does arithemetic operations on the inputs')
                    ] + state['messages']
                )
            ] , 
            'llm_calls' : state.get('llm_calls', 0) + 1

    }




def tool_node(state: MessageState):
    ''' Performs tool call '''

    results = []
    for tool_call in state['messages'][-1].tool_calls:
        tool = tools_by_name[tool_call['name']]
        observation = tool.invoke(tool_call['args'])
        results.append(ToolMessage(content=observation, tool_call_id = tool_call['id']))
        return {'messages' : results}


def should_continue(state: MessageState)-> Literal [END , "tool_node"]:
        """
            Checks if the graph can continue
        """

        if state['messages'][-1].tool_calls:
            return "tool_node"

        return END



agent_builder = StateGraph(MessageState)

agent_builder.add_node('llm_call',llm_call)
agent_builder.add_node('tool_node',tool_node)

agent_builder.add_edge(START, 'llm_call')
agent_builder.add_conditional_edges('llm_call' , should_continue, ["tool_node" ,END])
agent_builder.add_edge('tool_node' , 'llm_call')
agent=agent_builder.compile()

graph = agent.get_graph(xray=True)
png_bytes=graph.draw_mermaid_png()

with open('graph.png' , 'wb') as f:
    f.write(png_bytes)

messages = [HumanMessage(content="sum from 100 to 110")]
response = agent.invoke({'messages' : messages})

for r in response['messages']:
    r.pretty_print()


