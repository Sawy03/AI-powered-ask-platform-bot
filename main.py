from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retrieval

model = OllamaLLM(model="llama3.2:1b")

template = """
You are expert in answering questions based on the provided context.
You will be given a context and a question. Your task is to provide a concise answer based on the context.

Context: {context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n---------------------------------------------------------")
    user_input = input("Enter context and question : ")
    print("\n\n---------------------------------------------------------")
    context = retrieval.invoke(user_input)

    result = chain.invoke({"context": context, "question": user_input})
    print("\n\n---------------------------------------------------------")
    print("Context retrieved:")
    print(context)
    print("\n\n---------------------------------------------------------")
   
    print(result)
