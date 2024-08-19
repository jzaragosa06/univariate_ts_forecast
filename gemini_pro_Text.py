import pathlib
import textwrap

import google.generativeai as genai



from IPython.display import display
from IPython.display import Markdown

def to_markdown(text):
    text = text.replace(':', ' *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

model = genai.GenerativeModel('gemini-pro')

api_key = "AIzaSyDUFnIcM040z-zIN-d5EL4FGzOj_Ps5ybs"
genai.configure(api_key=api_key)


def explainForecastBehavior(behaviorRaw, context):
    response = model.generate_content(f"given this context {context}, explain the behavior of the time series forecast: {behaviorRaw}. Explain this in one paragraph")

    # print(to_markdown(response.text))
    # print(response.text)
    return response.text

def answerMessage(question, context, text_result):
    query = f"this is about time series forecast. given this context -- {context} -- and given the behavior of the forecast -- {text_result}. Answer this question -- {question}. answer this in one paragraph."
    response = model.generate_content(query)

    return response.text
