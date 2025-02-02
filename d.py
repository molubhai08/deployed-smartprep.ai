import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from crewai import Agent, Task, Crew
import pandas as pd
from io import StringIO
import re
from datetime import datetime
import os
import plotly.express as px
from sentence_transformers import SentenceTransformer

from typing import List

# Create a wrapper class for SentenceTransformer
class SentenceTransformerWrapper:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()
        
    def __call__(self, text: str | List[str]) -> List[float] | List[List[float]]:
        """Make the wrapper callable for single texts or lists of texts"""
        if isinstance(text, str):
            return self.embed_query(text)
        return self.embed_documents(text)

# Create the wrapper instance
embedding_model = SentenceTransformerWrapper("yandac/embedding_model_search_api")

# Now use it with FAISS


llm = ChatGroq(groq_api_key="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", model="llama3-70b-8192")

from crewai.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun


class MyCustomDuckDuckGoTool(BaseTool):
    name: str = "DuckDuckGo Search Tool"
    description: str = "Search the web for a given query."

    def _run(self, query: str) -> str:
        # Ensure the DuckDuckGoSearchRun is invoked properly.
        duckduckgo_tool = DuckDuckGoSearchRun()
        response = duckduckgo_tool.invoke(query)
        return response

    def _get_tool(self):
        # Create an instance of the tool when needed
        return MyCustomDuckDuckGoTool()
    
Duck_search = MyCustomDuckDuckGoTool()

agent6 = Agent(
    role='YouTube Educational Content Curator',
   goal='Search and curate relevant educational YouTube videos based on topics and provide direct video links',
   
   backstory="""An AI curator specialized in:
   - Educational video content search
   - Quality content verification
   - Relevant resource curation
   - Learning material organization""",
   
        llm="groq/llama3-70b-8192" , max_iter=5 ,tools= [Duck_search] ,
        verbose=True,
        allow_delegation=False
    )

def generate_url(topic):
    """
    Generate questions using the CrewAI agent for a specific topic
    """
    # Create a task for the agent
    quiz_generation_task = Task(
        description=f"""Research and give 5 urls for class 11 students on the topic: {topic}
        
        Search Requirements:
       - Validate link accessibility
       
       Output Format:
       - Video title
       - Channel name
       - Direct YouTube link
       - Make sure is a valid youtube link """,
        
        agent=agent6,
    
        expected_output="""4 urls on the topic provided"""
    )


    # Create a crew with the task
    crew = Crew(
        agents=[agent6],
        tasks=[quiz_generation_task]
    )

    # Execute the task
    result = crew.kickoff()
    return result.raw

agent7 = Agent(
    role='User Performance Diagnostics Specialist',
    goal="""Analyze incorrect questions and identify specific knowledge gaps, learning weaknesses, and improvement areas for users across different topics.""",
    backstory="""You are an advanced diagnostic agent specialized in educational performance analysis. Your primary objective is to:
    - Systematically evaluate user's incorrect responses
    - Identify precise knowledge gaps""",
    llm="groq/llama3-70b-8192",
    max_iter=3,
    verbose=True,
    allow_delegation=False
)

def weak_areas(paragraph, topic):
    performance_task = Task(
        description=f"""Analyze incorrect questions {paragraph} for topic: {topic}
        
        Analysis Requirements:
        - Identify specific knowledge gaps
        - Highlight weak areas in the concept
        
        Diagnostic Dimensions:
        - Conceptual Weakness Mapping
        
        Output Format:
        - Structured markdown report
        - Quantitative performance metrics
        
        Key Focus Areas:
        - Pattern of misconceptions
        - Difficulty level correlation
        - Topic-specific challenge areas""",
        
        agent=agent7,
        expected_output="""Comprehensive performance diagnostic report with identifying all the highlighting all the weak areas"""
    )

    crew = Crew(
        agents=[agent7],
        tasks=[performance_task]
    )

    # Execute the task
    result = crew.kickoff()
 # Pretty-printed string
    return result.raw

agent8 = Agent(
    role='Personalized Learning Strategy Developer',
    goal='Generate targeted, adaptive learning improvement strategies based on individual performance analysis',
    backstory="""An AI-powered learning optimization specialist focused on:
    - Suggest topics to work on 
    - Suggest difficulty levels to work on 
    - Suggest type of questions to work on
    - Designing customized improvement pathways
    - Recommending strategic learning interventions
    - Transforming weaknesses into learning opportunities""",
    llm="groq/llama3-70b-8192",
    max_iter=3,
    verbose=True
)

def generate_improvement_strategy(paragraph, topic):
    improvement_task = Task(
        description=f"""Generate actionable learning improvement strategy for the mistakes in questions with difficulty {paragraph} from {topic}
        
        Strategy Development Requirements:
        - Propose targeted improvement actions
        - Create structured learning roadmap
        
        Recommendation Dimensions:
        1. Topic-specific skill enhancement
        2. Question type practice strategies
        3. Difficulty level progression
        4. Conceptual understanding reinforcement
        """,
        
        agent=agent8,
        expected_output="""Comprehensive improvement strategy with:
        - Precise learning recommendations
        - Targeted practice suggestions
        - Skill development roadmap"""
    )

    crew = Crew(
        agents=[agent8],
        tasks=[improvement_task]
    )

    # Execute the task
    result = crew.kickoff()  # Pretty-printed string
    return result.raw

agent9 = Agent(
   role='Performance Analysis and Skill Profiler',
   goal='Conduct comprehensive user performance evaluation by identifying strengths, weaknesses, and learning potential',
   
   backstory="""Advanced performance diagnostic specialist focused on:
   - Detailed answer pattern analysis
   - Skill competency mapping
   - Personalized performance insights
   - Constructive feedback generation""",

   
   llm="groq/llama3-70b-8192",
   max_iter=3,
   verbose=True
)

def analyze_user_performance(correct_answers, wrong_answers , topic):
   """
   Generate comprehensive performance analysis
   """
   performance_task = Task(
       description=f"""Analyze user performance across answers with correct questions with difficulty as {correct_answers} and wrong questions with difficulty as {wrong_answers} for {topic}
       
       Analysis Dimensions:
       - Strengths and Weaknesses of the user
       - Answer pattern recognition
       - Skill competency mapping
       - Strength and weakness identification
       - Learning potential evaluation""",
       
       agent=agent9,
       expected_output=f"Detailed performance diagnostic report including the strength and weaknesses of the user in the topic {topic}"
   )




   crew = Crew(
        agents=[agent9],
        tasks=[performance_task]
    )

        # Execute the task
   result = crew.kickoff()
   return result.raw


# Streamlit app starts here


st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to bottom, rgb(30, 27, 75), #8080);
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Streamlit app starts here
st.title("SmartPrep.Ai ")


st.header("Add a username")
user_input = st.text_input("Enter your user name:")


if user_input :
    
    choice = st.selectbox("Select an option:", ["Youtube Link", "PDF Link"])

    text = ""

    if choice == "PDF Link":
    
    # Step 1: Upload Document
        st.header("Upload document for quiz creation")
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")


        if uploaded_file:
            temp_file_path = os.path.join("temp_uploaded_file.pdf")
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load and process document
            st.spinner("Processing the document...")
            loader = PyPDFLoader(temp_file_path)
            text = loader.load()

            os.remove(temp_file_path)
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final = splitter.split_documents(text)

            # Create embeddings and FAISS database
            db = FAISS.from_documents(final, embedding_model)
            retriever = db.as_retriever()

            # Define prompt and RAG chains
            retriever_prompt = ("""You are a specialized question generator for CSV-formatted assessment for the {context}.

        STRICT OUTPUT FORMAT:
        - Just output the asked questions, nothing else.
        - No headers, introductions, or explanatory text allowed
        - Produce ONLY  in CSV format.

        Question Generation Rules:
        1. Generate 10 unique questions in context of the topic
        2. Each question must:
        - Be distinct in wording
        - Cover different aspects of the topic
        - Avoid repetition
                                
        OUTPUT FORMAT: Generate questions in CSV format with columns: "question","optionA","optionB","optionC","optionD","correct","difficulty"

        STRICT CSV COLUMN REQUIREMENTS THAT NEEDS TO BE FOLLOWED:
        - question
        - optionA
        - optionB
        - optionC
        - optionD
        - correct 
        - difficulty

        Question Characteristics:
        - Test comprehensive understanding

        Difficulty Distribution:
        - 3 Easy, 4 Moderate and 3 Challenging questions

        Note:              
        NO EXTRA TEXT OR HEADLINES OR ENDING LINES
        DO NOT INCLUDE ANY HEADINGS OR EXTRA TEXTS
        Remember that you are a MCQ generating machine and you do not generate anything except that.

        Output Expectation:
        Precise, structured CSV questions with all the listed column names without any supplementary information
                                
        Example Output:
        
        "question","optionA","optionB","optionC","optionD","correct","difficulty"
        "What is the reason for the irregular variation of ionisation enthalpies in the 3d series?","Varying degree of stability of different 3d-configurations","Shielding effect of 3d electrons","Effective nuclear charge","Electron-electron repulsion","Varying degree of stability of different 3d-configurations","moderate"
        "Why do oxygen and fluorine tend to oxidise the metal to its highest oxidation state?","Due to their small size and high electronegativity","Due to their high reactivity","Due to their ability to form multiple bonds","Due to their high ionisation energy","Due to their small size and high electronegativity","easy"
        "What is the characteristic of the transition metals that is responsible for their ability to form coloured ions?","Incompletely filled d-orbitals","Completely filled d-orbitals","Incompletely filled f-orbitals","Completely filled f-orbitals","Incompletely filled d-orbitals","easy"
        "Why is Cr2+ a stronger reducing agent than Fe2+?","Because Cr2+ has a higher oxidation state","Because Cr2+ has a lower oxidation state","Because Cr2+ has a more stable electronic configuration","Because Cr2+ has a less stable electronic configuration","Because Cr2+ has a less stable electronic configuration","moderate"
        "What is the reason for the formation of complex compounds by transition metals?","Due to the availability of vacant d-orbitals","Due to the high reactivity of transition metals","Due to the ability of transition metals to form multiple bonds","Due to the low ionisation energy of transition metals","Due to the availability of vacant d-orbitals","challenging" """)
            
            prompt = ChatPromptTemplate.from_messages([("system", retriever_prompt), ("human", "{input}")])
            document_chain = create_stuff_documents_chain(llm, prompt)
            chain = create_retrieval_chain(retriever, document_chain)

            prompt_2 = ChatPromptTemplate.from_template(
            """Use ONLY the following context to answer the question. 
            If the answer is not in the context, say "I cannot find the answer in the provided document."

            Context:
            {context}

            Question: {input}

            """
        )   
            
            retriever_2 = db.as_retriever()

            document_chain_2 = create_stuff_documents_chain(llm ,prompt_2 )

            chain_2 = create_retrieval_chain(retriever_2 , document_chain_2)

            inpu = "what is this document about in 5 words or less"

            response = chain_2.invoke({"input" : inpu})

            topic = response['answer']


            # Step 2: Generate Test
        # Step 2: Generate Test
        # Step 2: Generate Test
            st.header("Generate Questions")
            if "test_generated" not in st.session_state:
                st.session_state.test_generated = False  # Track if the test is generated

            if st.button("Generate Test") or st.session_state.test_generated:
                st.session_state.test_generated = True
                if "df" not in st.session_state:  # Avoid regenerating if already generated
                    inpu = "Generate 10 questions from the file provided"
                    response = chain.invoke({"input": inpu})
                    csv_content = response['answer']
                    cleaned_content = re.sub(r"^.*generated questions in CSV format.*\n?", "", csv_content, flags=re.MULTILINE)
                    csv_data = StringIO(cleaned_content)
                    df = pd.read_csv(csv_data) 
                    st.session_state.df= df # Save the DataFrame in session_state
                st.write("Test generated successfully!")

            # Step 3: Take Test
            if st.session_state.test_generated:
                st.header("Take the Test")
                df = st.session_state.df  # Access the saved DataFrame
                if "answers" not in st.session_state:
                    st.session_state.answers = [None] * len(df)  # Placeholder for answers
                
                for i in range(df.shape[0]):

                    st.markdown(
        """
        <style>
        .highlight {
            font-size: 30px !important;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
        
    )               
                    st.markdown(f'<p class="highlight">Q{i+1}: {df['question'].iloc[i]}</p>', unsafe_allow_html=True)
                    # st.write(f"Q{i+1}: {df['question'].iloc[i]}")
                    
                    # Display radio buttons for answers
                    selected_answer = st.radio("",
                        options=[f"{df['optionA'].iloc[i]}", f"{df['optionB'].iloc[i]}", f"{df['optionC'].iloc[i]}", f"{df['optionD'].iloc[i]}"],
                        key=f"q{i}",
                        index=None
                    )
                    # Update session_state with the selected answer
                    st.session_state.answers[i] = selected_answer

                # Step 4: Evaluate Test
                if st.button("Submit Test"):
                    df['given'] = st.session_state.answers
                    wrong = []

                    for i in range(df.shape[0]):
                        if df['correct'].iloc[i] != df['given'].iloc[i]:
                            wrong.append("wrong")
                        else:
                            wrong.append("correct")

                    percentage = (wrong.count("correct") / df.shape[0]) * 100
                    correct_count = wrong.count("correct")
                    wrong_count = wrong.count("wrong")

                    # Display results

                    st.write(f"You scored {correct_count} out of {len(wrong)} ({percentage:.2f}%)")
                    l = ['correct' , 'wrong']
                    c1 = 0
                    c2 = 0
                    c3 = 0
                    w1 = 0
                    w2 = 0
                    w3 = 0
                    for i in range(len(wrong)):
                        if wrong[i] == 'wrong':
                            if df['difficulty'].iloc[i] == 'easy':
                                w1 = w1 + 1 
                            elif df['difficulty'].iloc[i] == 'moderate':
                                w2 = w2 + 1 
                            else:
                                w3 = w3 + 1 
                        else:
                            if df['difficulty'].iloc[i] == 'easy':
                                c1 = c1 + 1 
                            elif df['difficulty'].iloc[i] == 'moderate':
                                c2 = c2 + 1 
                            else:
                                c3 = c3 + 1 
                    easy = [c1 , w1]
                    medium = [c2 , w2]
                    diff = [c3 , w3]
                    
                    dat = {
        'Easy': easy, 'Medium':medium , 'Difficulty':diff ,
        'Correct/Wrong': l}
                    data = pd.DataFrame(dat)
                    df_melted = data.melt(id_vars='Correct/Wrong', value_vars=['Easy', 'Medium' ,'Difficulty' ])
                    fig = px.bar(df_melted, x='Correct/Wrong', y='value',color='variable', barmode='group', title='Analytics of wrong and right questions' )

                    st.plotly_chart(fig)
                    st.write("### Incorrect Questions:")

                    for i in range(len(wrong)):
                        if wrong[i] == "wrong":
                            st.write(f"Q: {df['question'].iloc[i]}")
                            st.write(f"Correct Answer: {df['correct'].iloc[i]}, Your Answer: {df['given'].iloc[i]}")

                    # Save results in the database
                    today = datetime.now().strftime("%Y-%m-%d")
                    table_name = f"{user_input}_result"

    # Correct SQL query



                    # Modify the button to toggle the state
                if st.button("Analyze Weak Areas"):
                    st.session_state.analyze_clicked = True

                if st.session_state.get("analyze_clicked", False):
                    df['given'] = st.session_state.answers
                    wrong = []

                    for i in range(df.shape[0]):
                        if df['correct'].iloc[i] != df['given'].iloc[i]:
                            wrong.append("wrong")
                        else:
                            wrong.append("correct")
                    paragraph = ". ".join([
                        f"{df['question'].iloc[i]} (Difficulty: {df['difficulty'].iloc[i]})"
                        for i in range(len(wrong)) if wrong[i] == 'wrong'
                    ])
                    if not paragraph.strip():
                        st.error("No incorrect questions found to analyze.")
                    else:
                        with st.spinner("Analyzing weak areas..."):
                            try:
                                result = weak_areas(paragraph, topic)
                                st.success("Analysis Complete!")
                                st.write("Weak Area Analysis:")
                                st.write(result)
                            except Exception as e:
                                st.error(f"Error during analysis: {e}")


                # Generate Improvement Strategy Button
                if st.button("Generate Improvement Strategy"):
                    st.session_state.strategy_clicked = True

                if st.session_state.get("strategy_clicked", False):
                    paragraph = ". ".join([
                        f"{df['question'].iloc[i]} (Difficulty: {df['difficulty'].iloc[i]})"
                        for i in range(len(wrong)) if wrong[i] == 'wrong'
                    ])
                    if not paragraph.strip():
                        st.error("No incorrect questions found to generate strategy.")
                    else:
                        with st.spinner("Generating improvement strategy..."):
                            try:
                                result = generate_improvement_strategy(paragraph, topic)
                                st.success("Strategy Generated!")
                                st.write("Improvement Strategy:")
                                st.write(result)
                            except Exception as e:
                                st.error(f"Error during strategy generation: {e}")

                if st.button("Weakness and Strength"):
                    st.session_state.wc_clicked = True

                if st.session_state.get("wc_clicked", False):
                    paragraph = ". ".join([
                        f"{df['question'].iloc[i]} (Difficulty: {df['difficulty'].iloc[i]})"
                        for i in range(len(wrong)) if wrong[i] == 'wrong'
                    ])
                    
                    right= ". ".join([
                        f"{df['question'].iloc[i]} (Difficulty: {df['difficulty'].iloc[i]})"
                        for i in range(len(wrong)) if wrong[i] == 'correct'
                    ])

                    if not paragraph.strip():
                        st.error("No incorrect questions found to generate weakness.")
                    else:
                        with st.spinner("Generating improvement strategy..."):
                            try:
                                result = analyze_user_performance(right , paragraph, topic)
                                st.success("Strength/Weakness Generated!")
                                st.write("Strength/Weakness:")
                                st.write(result)
                            except Exception as e:
                                st.error(f"Error during strategy generation: {e}")


                if st.button("Recommend Videos for the Mistakes"):
                    st.session_state.videos_clicked = True

                if st.session_state.get("videos_clicked", False):
                    
                    with st.spinner("Finding URL'S..."):
                        try:
                            result = generate_url(topic)
                            st.success("Channels and URL found!!!")
                            st.write(f"Videos on the topic {topic}")
                            st.write(result)

                        except Exception as e:
                            st.error(f"Error video finding: {e}")


    else:
        st.header("Upload video for quiz creation")

        user_in = st.text_input("Enter your link ")

        if user_in:
            loader = YoutubeLoader.from_youtube_url(user_in)
            text = loader.load()


            # Split text into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final = splitter.split_documents(text)

            # Create embeddings and FAISS database
            db = FAISS.from_documents(final, embedding_model)
            retriever = db.as_retriever()

            # Define prompt and RAG chains
            retriever_prompt = ("""You are a specialized question generator for CSV-formatted assessment for the {context}.

        STRICT OUTPUT FORMAT:
        - Just output the asked questions, nothing else.
        - No headers, introductions, or explanatory text allowed
        - Produce ONLY  in CSV format.

        Question Generation Rules:
        1. Generate 10 unique questions in context of the topic
        2. Each question must:
        - Be distinct in wording
        - Cover different aspects of the topic
        - Avoid repetition
                                
        OUTPUT FORMAT: Generate questions in CSV format with columns: "question","optionA","optionB","optionC","optionD","correct","difficulty"

        STRICT CSV COLUMN REQUIREMENTS THAT NEEDS TO BE FOLLOWED:
        - question
        - optionA
        - optionB
        - optionC
        - optionD
        - correct 
        - difficulty

        Question Characteristics:
        - Test comprehensive understanding

        Difficulty Distribution:
        - 3 Easy, 4 Moderate and 3 Challenging questions

        Note:              
        NO EXTRA TEXT OR HEADLINES OR ENDING LINES
        DO NOT INCLUDE ANY HEADINGS OR EXTRA TEXTS
        Remember that you are a MCQ generating machine and you do not generate anything except that.

        Output Expectation:
        Precise, structured CSV questions with all the listed column names without any supplementary information
                                
        Example Output:
        "question","optionA","optionB","optionC","optionD","correct","difficulty"
        "What is the proper procedure when approaching a blind curve on a wet road?","Maintain current speed but move to the center of lane","Increase speed slightly to maintain momentum","Reduce speed and move to the right side of lane","Apply brakes firmly while straightening the bike","Reduce speed and move to the right side of lane","moderate" """)
            
            prompt = ChatPromptTemplate.from_messages([("system", retriever_prompt), ("human", "{input}")])
            document_chain = create_stuff_documents_chain(llm, prompt)
            chain = create_retrieval_chain(retriever, document_chain)

            prompt_2 = ChatPromptTemplate.from_template(
            """Use ONLY the following context to answer the question. 
            If the answer is not in the context, say "I cannot find the answer in the provided document."

            Context:
            {context}

            Question: {input}

            """
        )   
            
            retriever_2 = db.as_retriever()

            document_chain_2 = create_stuff_documents_chain(llm ,prompt_2 )

            chain_2 = create_retrieval_chain(retriever_2 , document_chain_2)

            inpu = "what is this document about in 5 words or less"

            response = chain_2.invoke({"input" : inpu})

            topic = response['answer']


            # Step 2: Generate Test
        # Step 2: Generate Test
        # Step 2: Generate Test
            st.header("Generate Questions")
            if "test_generated" not in st.session_state:
                st.session_state.test_generated = False  # Track if the test is generated

            if st.button("Generate Test") or st.session_state.test_generated:
                st.session_state.test_generated = True
                if "df" not in st.session_state:  # Avoid regenerating if already generated
                    inpu = "Generate 10 questions from the file provided"
                    response = chain.invoke({"input": inpu})
                    csv_content = response['answer']
                    cleaned_content = re.sub(r"^.*generated questions in CSV format.*\n?", "", csv_content, flags=re.MULTILINE)
                    csv_data = StringIO(cleaned_content)
                    df = pd.read_csv(csv_data) 
                    st.session_state.df= df # Save the DataFrame in session_state
                st.write("Test generated successfully!")

            # Step 3: Take Test
            if st.session_state.test_generated:
                st.header("Take the Test")
                df = st.session_state.df  # Access the saved DataFrame
                if "answers" not in st.session_state:
                    st.session_state.answers = [None] * len(df)  # Placeholder for answers
                
                for i in range(df.shape[0]):

                    st.markdown(
        """
        <style>
        .highlight {
            font-size: 30px !important;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
        
    )               
                    st.markdown(f'<p class="highlight">Q{i+1}: {df['question'].iloc[i]}</p>', unsafe_allow_html=True)
                    # st.write(f"Q{i+1}: {df['question'].iloc[i]}")
                    
                    # Display radio buttons for answers
                    selected_answer = st.radio("",
                        options=[f"{df['optionA'].iloc[i]}", f"{df['optionB'].iloc[i]}", f"{df['optionC'].iloc[i]}", f"{df['optionD'].iloc[i]}"],
                        key=f"q{i}",
                        index=None
                    )
                    # Update session_state with the selected answer
                    st.session_state.answers[i] = selected_answer

                # Step 4: Evaluate Test
                if st.button("Submit Test"):
                    df['given'] = st.session_state.answers
                    wrong = []

                    for i in range(df.shape[0]):
                        if df['correct'].iloc[i] != df['given'].iloc[i]:
                            wrong.append("wrong")
                        else:
                            wrong.append("correct")

                    percentage = (wrong.count("correct") / df.shape[0]) * 100
                    correct_count = wrong.count("correct")
                    wrong_count = wrong.count("wrong")

                    # Display results

                    st.write(f"You scored {correct_count} out of {len(wrong)} ({percentage:.2f}%)")
                    l = ['correct' , 'wrong']
                    c1 = 0
                    c2 = 0
                    c3 = 0
                    w1 = 0
                    w2 = 0
                    w3 = 0
                    for i in range(len(wrong)):
                        if wrong[i] == 'wrong':
                            if df['difficulty'].iloc[i] == 'easy':
                                w1 = w1 + 1 
                            elif df['difficulty'].iloc[i] == 'moderate':
                                w2 = w2 + 1 
                            else:
                                w3 = w3 + 1 
                        else:
                            if df['difficulty'].iloc[i] == 'easy':
                                c1 = c1 + 1 
                            elif df['difficulty'].iloc[i] == 'moderate':
                                c2 = c2 + 1 
                            else:
                                c3 = c3 + 1 
                    easy = [c1 , w1]
                    medium = [c2 , w2]
                    diff = [c3 , w3]
                    
                    dat = {
        'Easy': easy, 'Medium':medium , 'Difficulty':diff ,
        'Correct/Wrong': l}
                    data = pd.DataFrame(dat)
                    df_melted = data.melt(id_vars='Correct/Wrong', value_vars=['Easy', 'Medium' ,'Difficulty' ])
                    fig = px.bar(df_melted, x='Correct/Wrong', y='value',color='variable', barmode='group', title='Analytics of wrong and right questions' )

                    st.plotly_chart(fig)
                    st.write("### Incorrect Questions:")


                    for i in range(len(wrong)):
                        if wrong[i] == "wrong":
# 
                            st.write(f"Q: {df['question'].iloc[i]}")
                            st.write(f"Correct Answer: {df['correct'].iloc[i]}, Your Answer: {df['given'].iloc[i]}")

                    # Save results in the database
                    today = datetime.now().strftime("%Y-%m-%d")
                    table_name = f"{user_input}_result"



                    # Modify the button to toggle the state
                if st.button("Analyze Weak Areas"):
                    st.session_state.analyze_clicked = True

                if st.session_state.get("analyze_clicked", False):
                    df['given'] = st.session_state.answers
                    wrong = []

                    for i in range(df.shape[0]):
                        if df['correct'].iloc[i] != df['given'].iloc[i]:
                            wrong.append("wrong")
                        else:
                            wrong.append("correct")
                    paragraph = ". ".join([
                        f"{df['question'].iloc[i]} (Difficulty: {df['difficulty'].iloc[i]})"
                        for i in range(len(wrong)) if wrong[i] == 'wrong'
                    ])
                    if not paragraph.strip():
                        st.error("No incorrect questions found to analyze.")
                    else:
                        with st.spinner("Analyzing weak areas..."):
                            try:
                                result = weak_areas(paragraph, topic)
                                st.success("Analysis Complete!")
                                st.write("Weak Area Analysis:")
                                st.write(result)
                            except Exception as e:
                                st.error(f"Error during analysis: {e}")


                # Generate Improvement Strategy Button
                if st.button("Generate Improvement Strategy"):
                    st.session_state.strategy_clicked = True

                if st.session_state.get("strategy_clicked", False):
                    paragraph = ". ".join([
                        f"{df['question'].iloc[i]} (Difficulty: {df['difficulty'].iloc[i]})"
                        for i in range(len(wrong)) if wrong[i] == 'wrong'
                    ])
                    if not paragraph.strip():
                        st.error("No incorrect questions found to generate strategy.")
                    else:
                        with st.spinner("Generating improvement strategy..."):
                            try:
                                result = generate_improvement_strategy(paragraph, topic)
                                st.success("Strategy Generated!")
                                st.write("Improvement Strategy:")
                                st.write(result)
                            except Exception as e:
                                st.error(f"Error during strategy generation: {e}")

                if st.button("Weakness and Strength"):
                    st.session_state.wc_clicked = True

                if st.session_state.get("wc_clicked", False):
                    paragraph = ". ".join([
                        f"{df['question'].iloc[i]} (Difficulty: {df['difficulty'].iloc[i]})"
                        for i in range(len(wrong)) if wrong[i] == 'wrong'
                    ])
                    
                    right= ". ".join([
                        f"{df['question'].iloc[i]} (Difficulty: {df['difficulty'].iloc[i]})"
                        for i in range(len(wrong)) if wrong[i] == 'correct'
                    ])

                    if not paragraph.strip():
                        st.error("No incorrect questions found to generate weakness.")
                    else:
                        with st.spinner("Generating improvement strategy..."):
                            try:
                                result = analyze_user_performance(right , paragraph, topic)
                                st.success("Strength/Weakness Generated!")
                                st.write("Strength/Weakness:")
                                st.write(result)
                            except Exception as e:
                                st.error(f"Error during strategy generation: {e}")


                if st.button("Recommend Videos for the Mistakes"):
                    st.session_state.videos_clicked = True

                if st.session_state.get("videos_clicked", False):
                    
                    with st.spinner("Finding URL'S..."):
                        try:
                            result = generate_url(topic)
                            st.success("Channels and URL found!!!")
                            st.write(f"Videos on the topic {topic}")
                            st.write(result)

                        except Exception as e:
                            st.error(f"Error video finding: {e}")

  

