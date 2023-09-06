streamlit_app.py




import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from hugchat import hugchat

import os

import random

import uuid

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import create_csv_agent
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType

from langchain.schema import AIMessage, HumanMessage, SystemMessage

from langchain.llms import OpenAI

from sqlalchemy.sql import text

st.set_page_config(page_title="Parent ChatBot")

"""
# Parent ChatBot
#### ABAiGuide 💬 (Beta)

"""

# Sidebar contents
# Generate empty lists for generated and past.
## generated stores AI generated responses
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hi I'm the Parent Chatbot, how may I help you?"]
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']

# Convert the chat history into a text format
def chat_to_text(past, generated):
    chat_text = ""
    for i in range(len(past)):
        chat_text += f"User: {past[i]}\n"
        chat_text += f"Parent Chatbot: {generated[i]}\n\n"
    return chat_text

st.sidebar.title('ABAiGuide 💬 (Beta)')
openai_api_key = ''
if len(os.environ['OPENAI_API_KEY']) > 0:
    openai_api_key = os.environ['OPENAI_API_KEY']
else:
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
model_name = st.sidebar.radio("Model", ["gpt-4", "gpt-3.5-turbo"], horizontal=True)
st.sidebar.markdown("Note. GPT-4 is recommended for better performance.")
st.sidebar.markdown('''
## About
This is a Beta version/prototype of an AI powered app providing behavioral support for parents and families of children with Autism and related developmental disorders. The app combines the principles of Applied Behavior Analysis, with the LLM GPT 3.5 / 4 to create a chatbot for parents to ask questions about their child's behavior and get specific actionable recommendations.

Haley McPeek, MS, BCBA
''')

app_env = st.secrets["env_vars"]["app_env"]
db_name = f"postgresql_{app_env}"
print('db_name: ')
print(db_name)
conn = st.experimental_connection(db_name, type='sql')
with conn.session as s:
    s.execute(text('''
    CREATE TABLE IF NOT EXISTS conversations
    (
        conv_id varchar(80) NOT NULL,
        time_stamp timestamp default NULL,
        model_name varchar(80) default NULL,
        dialog_turn integer,
        user_text text default NULL,
        response_text text default NULL
    )
    '''))
    s.commit()

if 'session_id' not in st.session_state:
    st.session_state['session_id'] = uuid.uuid1()
session_id = st.session_state['session_id']
print('session_id: ')
print(session_id)

if 'dialog_turn' not in st.session_state:
    st.session_state['dialog_turn'] = 0

# Provide a button to download the chat history
st.sidebar.download_button(
    label="Download chat history",
    data=chat_to_text(st.session_state['past'], st.session_state['generated']),
    file_name="chat_history.txt",
    mime="text/plain"
)

# Layout of input/response containers
response_container = st.container()
colored_header(label='', description='', color_name='blue-30')
input_container = st.container()

if 'input_buffer' not in st.session_state:
    st.session_state.input_buffer = ''

def submit_input():
    st.session_state.input_buffer = st.session_state.input
    st.session_state.input = ''

# User input
## Function for taking user provided prompt as input
def get_text():
    input_text = st.text_input("User: ", "", key="input", on_change=submit_input)
    return st.session_state.input_buffer

default_machine_prompt = """
You are an AI behavior analyst in our parent chat bot, responsible for analyzing and providing guidance on challenging behaviors exhibited by children. Your role is to assist parents in understanding the function and underlying causes of the behaviors, as well as offer strategies and recommendations to address them effectively.
After gathering information about the behavior and providing initial recommendations, consider if a personalized social story could be beneficial for the child. If you believe it would be helpful, offer to create a social story for the parent and child.
To personalize the social story, you will need to gather specific information about the child's physical appearance. Ask the parent about the child's age, race/ethnicity, physical characteristics (e.g., hair color and style, eye color, any glasses or accessories), and clothing preferences. These details will allow you to generate accurate and relatable images for the social story, whether they are photorealistic or cartoon-like, based on the parent's preference.
Follow these guidelines to fulfill your role:

    1.  Information Gathering: Engage in a conversation with the parent and ask follow-up questions to gather additional details about the behavior. Ask only one question at a time until you have all needed information. Prompt the parent to provide a comprehensive description of the behavior, including antecedents, consequences, and any relevant context or factors that may influence it.
    2.  Behavior Analysis: Based on the information provided, analyze the behavior using the principles of Applied Behavior Analysis (ABA). Consider the antecedents, consequences, and potential functions of the behavior to determine its underlying causes.
    3.  Functional Analysis: Identify the most likely function(s) of the behavior by exploring its relationships with the environment and its impact on the child. Use your knowledge of ABA principles to understand the underlying motivations and factors contributing to the behavior.
    4.  Evidence-Based Strategies: Draw from evidence-based strategies rooted in ABA principles to guide parents in addressing the behavior. Tailor the recommendations to the specific function of the behavior and the child’s individual needs and abilities.
    5.  Parent-Friendly Language: Communicate the analysis and recommendations in clear and concise language that parents can easily understand. Use plain language and provide practical examples or scenarios to help parents apply the strategies effectively.
    6.  Empowerment and Support: Empower parents by acknowledging their expertise as primary caregivers and encourage their active involvement in implementing behavior management strategies. Provide support and guidance to build their confidence and ensure a positive and supportive environment for their child.
    7.  Continuous Learning: Stay updated on the latest research and advancements in behavior analysis to enhance your knowledge and expertise. Incorporate new insights and evidence-based practices into your recommendations to provide the most accurate and effective guidance.

Remember, your role is not only to analyze the behavior but also to provide personalized and effective recommendations, potentially including the creation of a custom social story.

Also remember to only ask one follow up question at a time. You will have time to ask more follow up questions in the future.

"""

def generate_response(chat, machine_prompt, human_prompt, session_state):
    system_prompt = f"""
    {machine_prompt}

    These are the previous dialogs in the conversation:

    """
    for i, (past, generated) in enumerate(zip(session_state.past, session_state.generated)):
        previous_conv = f"""
            dialog_turn: {i}
            user: {past}
            parent_chatbot: {generated}

        """
        system_prompt += previous_conv

    print("system_prompt: ")
    print(system_prompt)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]
    return chat(messages).content

if len(openai_api_key) > 0:

    st.divider()

    ## Set OpenAI API Key (get from https://platform.openai.com/account/api-keys)
    os.environ["OPENAI_API_KEY"] = openai_api_key

    ## Instantiate model
    llm = ChatOpenAI(model_name=model_name, temperature=0.5)

    ## Applying the user input box
    with input_container:
        user_input = get_text()

    ## Conditional display of AI generated responses as a function of user provided prompts
    with response_container:
        if user_input:
            response = generate_response(llm, default_machine_prompt, user_input, st.session_state)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(response)
            # Log the response
            with conn.session as s:
                s.execute(text(
                    'INSERT INTO conversations (conv_id, time_stamp, model_name, dialog_turn, user_text, response_text) VALUES (:conv_id, CURRENT_TIMESTAMP, :model_name, :dialog_turn, :user_text, :response_text);'),
                    params=dict(conv_id=session_id, model_name=model_name, dialog_turn = st.session_state['dialog_turn'], user_text=user_input, response_text=response)
                )
                s.commit()
            st.session_state['dialog_turn'] += 1
            
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))
