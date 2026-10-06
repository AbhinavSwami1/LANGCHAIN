import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


## page configuration
st.set_page_config(
    page_title="LangChain Chatbot with Groq",
    page_icon="🤖"
)


#title
st.title(" 🤖 Simple LangChain Chatbot with Groq")
st.markdown(
    "This is a simple chatbot built using LangChain and Groq. "
    "You can ask questions and get responses from the model."
)


with st.sidebar:
    st.header("Settings")

    #api key 
    api_key = st.text_input(
        "Enter your Groq API Key",
        type="password",
        key="api_key",
        help="You can get your API key from the Groq developer portal."
    )

    # model selection
    model_name = st.selectbox(
        "Select Model",
        ["openai/gpt-oss-120b"],
        index=0,
        help="Select the model you want to use for the chatbot."
    )

    ## clear button

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.success("Chat history cleared!")
        st.rerun()


#initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# initialize llm model

@st.cache_resource
def get_chain(api_key, model_name):

    if not api_key:
        return None

    #initialize the chat model
    llm = ChatGroq(
        api_key=api_key,
        model=model_name,
        temperature=0.7,
        streaming=True
    )

    # create prompt template
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant powered by Groq. "
            "Answer the questions as accurately as possible."
        ),
        (
            "user",
            "{question}"
        )
    ])

    # create chain
    chain = prompt | llm | StrOutputParser()

    return chain


# get chain
chain = get_chain(api_key, model_name)


if not chain:

    st.warning(
        "Please enter your Groq API Key to initialize the chatbot."
    )

    st.markdown(
        "You can get your API key from the "
        "[Groq developer portal](https://console.groq.com/)."
    )

    st.stop()


else:

    #displaye messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.write(message["content"])


    # chat input

    if question := st.chat_input("Ask a question!!!"):

        # add user message to session state
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })


        # display user message
        with st.chat_message("user"):
            st.write(question)


        ## generate response from model
        with st.chat_message("assistant"):

            message_placeholder = st.empty()
            full_response = ""

            try:

                #streaming response from model
                for chunk in chain.stream({
                    "question": question
                }):

                    full_response += chunk

                    message_placeholder.markdown(
                        full_response + "▌"
                    )


                message_placeholder.markdown(full_response)


                #add to history

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })


            except Exception as e:

                st.error(
                    f"Error generating response: {e}"
                )