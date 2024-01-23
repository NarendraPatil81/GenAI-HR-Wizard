from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
import os
import streamlit as st
import openai
import os
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
import re
import pandas as pd
from datetime import datetime
from io import BytesIO
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import whisper
from streamlit_chat import message
from streamlit_option_menu import option_menu
import google.generativeai as genai
import pickle
import random
import tempfile
from langchain.text_splitter import CharacterTextSplitter
from PIL import Image
global docs
docs=[]
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
os.environ['GOOGLE_API_KEY'] = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
os.environ['GOOGLE_API_KEY'] = os.environ.get("GOOGLE_API_KEY")
if "first_time" not in st.session_state:
            st.session_state.first_time = False
generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

if st.session_state["first_time"]==False:
    with open(r"C:\Users\admin\Downloads\GenAI\bge.pkl", 'rb') as file:
        loaded_model = pickle.load(file)
        st.session_state["loaded_model"]=loaded_model
    model_ge = genai.GenerativeModel(model_name="gemini-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)
    st.session_state["model_ge"]=model_ge
    st.session_state["first_time"]=True
    
import fitz  # PyMuPDF

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

def extract_text_from_pdf(file_path,candidate_name):
    try:
        # Open the PDF file
        pdf_document = fitz.open(file_path)

        # Initialize an empty string to store the text content
        text_content = ""

        # Iterate through all pages
        for page_number in range(pdf_document.page_count):
            # Get the page
            page = pdf_document[page_number]

            # Extract text from the page
            text_content += page.get_text()

        # Return an instance of the Document class
        result = Document(
            page_content=text_content,
            metadata={
                'candidate name': candidate_name,
                'total_pages': pdf_document.page_count
            }
        )

        return result

    except Exception as e:
        result = Document(
            page_content=f"Error: {str(e)}",
            metadata={
                'source': file_path,
                'total_pages': 0
            }
        )
        return result

global Resume_Flag
if "prompt" not in st.session_state:
            st.session_state.prompt = []
if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
if "chat_ans_history" not in st.session_state:
            st.session_state.chat_ans_history = []
if "vectorstore_jd" not in st.session_state:
            st.session_state.vectorstore_jd = []
if "vectorstore" not in st.session_state:
            st.session_state.vectorstore_jd = []
if "text_chunks" not in st.session_state:
            st.session_state.text_chunks = []
if "docs" not in st.session_state:
            st.session_state.docs = []
if "loaded_model" not in st.session_state:
            st.session_state.loaded_model = []
if "model_ge" not in st.session_state:
            st.session_state.model_ge = []

st.set_page_config(page_title="HR Wizard")



# Hide Streamlit footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)




def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    loaded_model=st.session_state["loaded_model"]
    vectorstore = FAISS.from_documents(text_chunks,loaded_model)
    return vectorstore



def eval(): 
    st.title("🔍 Decode Interview Performance")
    text_feed =''
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    uploaded_files = st.file_uploader("Upload MP4 Files", type=["mp4"], accept_multiple_files=True)
    if uploaded_files and st.button("Evaluate"):
        st.header("Question and Answer with Score")
        for resume in uploaded_files:
            print(resume)
            print(resume.name)
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(resume.read())
            temp_file.close()
            video_path = "uploaded_video.mp4"
            model = whisper.load_model("base")
            transcript = model.transcribe(temp_file.name)
            transcript=transcript['text']
            print(transcript)
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
            doc = text_splitter.split_text(transcript)
            # chain = load_qa_chain(llm=OpenAI(), chain_type="map_reduce")
            python_docs = text_splitter.create_documents([transcript])
            que_ans =''
            st_1 =''
            before_overall_feedback_l = []
            after_overall_feedback_l=[]
            for i in doc:
                prompt_load =f'''
                Given the provided text{i} batch, your task is to extract all questions and answers and then assign a score out of 10 for each answer. The scoring should consider the following criteria:

                HR Questions: 
                For HR-related questions, evaluate how accurately the candidate responds. A comprehensive and relevant answer should receive a higher score.

                Technical Questions: 
                For technical questions, assess the correctness and depth of the candidate's response. An accurate and detailed answer should be awarded a higher score.

                Overall Feedback: 
                At the end, provide an overall assessment of the candidate's performance. Highlight strengths, areas of improvement, and any noteworthy observations.

                Remember, the scoring should be fair and impartial, reflecting the candidate's knowledge, communication skills, and suitability for the role. Provide constructive feedback to help guide the evaluation process.

                Your response should be well-organized and structured, clearly presenting the extracted questions and answers along with the assigned scores. 
                Avoid numbering the questions.



                Question:
                Answer: [Candidate's Answer]
                Score: [Score out of 10]
                Feedback:[Indicates whether improvement is needed or the correctness of the answer]

                Question:
                Answer: [Candidate's Answer]
                Score: [Score out of 10]
                Feedback:[Indicates whether improvement is needed or the correctness of the answer]

                Overall Feedback:
                - [Positive feedback]
                - [Negative feedback]
                - [Areas for improvement]
                - [Other observations]


                '''
                prompt_parts = [prompt_load]
                model_ge =st.session_state["model_ge"]
                response = model_ge.generate_content(prompt_parts)
                # completions = openai.Completion.create (engine="text-davinci-003",prompt=prompt_load,max_tokens=2200,n=1,stop=None,temperature=0.5,)
                # message = completions.choices[0].text
                message = response.text
                print(message)
                st_1 += message
                split_text = message.split("Overall Feedback", 1)
                before_overall_feedback = split_text[0]
                after_overall_feedback = split_text[1]
                # before_overall_feedback = re.search(r"(.+?)\nOverall Feedback:", message, re.DOTALL)
                # after_overall_feedback = re.search(r"(?<=Overall Feedback:\n)(.+)", message, re.DOTALL)

                if before_overall_feedback:
                    #extracted_text_before = before_overall_feedback.group(1)
                    before_overall_feedback_l.append(before_overall_feedback)
                    print("Extracted Text Before Overall Feedback:\n", before_overall_feedback)
                else:
                    print("Overall Feedback section not found.")
                for text in before_overall_feedback_l:
                    st.text(text)
                if after_overall_feedback:
                    extracted_text_after = after_overall_feedback
                    after_overall_feedback_l.append(extracted_text_after)
                    print("Extracted Text After Overall Feedback:\n", extracted_text_after)
                else:
                    print("Overall Feedback section not found.")
                st.header("Overall Feedback")
                for text in after_overall_feedback_l:
                    st.text(text)
        with open("candidate_evaluation_feedback.txt", "w") as file:
            for before_text, after_text in zip(before_overall_feedback_l, after_overall_feedback_l):
                text_feed =text_feed + "Question and Answers:\n"
                text_feed =text_feed + before_text + "\n\n"
                text_feed =text_feed + "'Overall Feedback':\n"
                text_feed =text_feed + after_text + "\n\n"
        filename = "interview.txt"
        text_bytes = text_feed.encode('utf-8')
        st.download_button(label="Download The Feedback", data=text_bytes, file_name=filename, mime='text/plain')

            
def extract_resume_info(resume_info_string):
    fields_list = []
    resume_info_dict = {
        "Name": "",
        "Job Profile": "",
        "Skill Set": "",
        "Email": "",
        "Phone Number": "",
        "Number of Years of Experience": "",
        "Previous Organizations and Technologies Worked With": "",
        "Education": "",
        "Certifications": "",
        "Projects": "",
        "Location": ""
    }

    # Use separate regular expressions for each field to capture their values.
    name_match = re.search(r'Name:\s(.*?)(?:\n|$)', resume_info_string)
    if name_match:
        resume_info_dict["Name"] = name_match.group(1).strip()

    job_profile_match = re.search(r'Job Profile:\s(.*?)(?:\n|$)', resume_info_string)
    if job_profile_match:
        resume_info_dict["Job Profile"] = job_profile_match.group(1).strip()

    skill_set_match = re.search(r'Skill Set:\s(.*?)(?:\n|$)', resume_info_string)
    if skill_set_match:
        resume_info_dict["Skill Set"] = skill_set_match.group(1).strip()

    email_match = re.search(r'Email:\s(.*?)(?:\n|$)', resume_info_string)
    if email_match:
        resume_info_dict["Email"] = email_match.group(1).strip()

    phone_number_match = re.search(r'Phone Number:\s(.*?)(?:\n|$)', resume_info_string)
    if phone_number_match:
        resume_info_dict["Phone Number"] = phone_number_match.group(1).strip()

    years_of_experience_match = re.search(r'Number of Years of Experience:\s(.*?)(?:\n|$)', resume_info_string)
    if years_of_experience_match:
        resume_info_dict["Number of Years of Experience"] = years_of_experience_match.group(1).strip()

    org_and_tech_match = re.search(r'Previous Organizations and Technologies Worked With:\s(.*?)(?:\n|$)', resume_info_string)
    if org_and_tech_match:
        resume_info_dict["Previous Organizations and Technologies Worked With"] = org_and_tech_match.group(1).strip()

    education_match = re.search(r'Education:\s(.*?)(?:\n|$)', resume_info_string)
    if education_match:
        resume_info_dict["Education"] = education_match.group(1).strip()

    certifications_match = re.search(r'Certifications:\s(.*?)(?:\n|$)', resume_info_string)
    if certifications_match:
        resume_info_dict["Certifications"] = certifications_match.group(1).strip()

    projects_match = re.search(r'Projects:\s(.*?)(?:\n|$)', resume_info_string)
    if projects_match:
        resume_info_dict["Projects"] = projects_match.group(1).strip()

    location_match = re.search(r'Location:\s(.*?)(?:\n|$)', resume_info_string)
    if location_match:
        resume_info_dict["Location"] = location_match.group(1).strip()

    #fields_list.append(resume_info_dict)

    return resume_info_dict

        

def preprocessing():
    with st.form("upload_files"):
        pdf_docs = st.file_uploader("Upload Resumes", type=["pdf"],accept_multiple_files=True)
        btn =st.form_submit_button("Upload")
        if btn:
            with st.spinner("Processing the files"):
                for uploaded_file in pdf_docs:
                    temp_file = tempfile.NamedTemporaryFile(delete=False)
                    temp_file.write(uploaded_file.read())
                    temp_file.close()
                    loader = extract_text_from_pdf(temp_file.name,uploaded_file.name)
                    docs.append(loader)
                global text_chunks
                text_chunks = get_text_chunks(docs)
                global vectorstore
                vectorstore = get_vectorstore(text_chunks)
                vectorstore_jd = get_vectorstore(docs)
                st.session_state["vectorstore"]=vectorstore
                st.session_state["vectorstore_jd"]=vectorstore_jd
                st.session_state["text_chunks"]=text_chunks
                st.session_state["docs"]=docs

                    

def rss():
    st.header("🔍 GenAI-Powered Resume Search Chatbot: Finding the Perfect Fit")
    vectorstore = st.session_state["vectorstore"]
    user_question = st.text_input("What type of information are you looking for in these resumes? Enter keywords or skills.")
    if user_question:
        with st.spinner("Processing"):
            k = min(5,len(st.session_state["docs"]))
            st.write(k)
            similar_docs = vectorstore.similarity_search(user_question,k=k)
            prompt_template = f"""
            I will provide a question along with a set of resumes. 
            Your task is to extract relevant information from these resumes to generate a concise answer. 
            If the exact information is not available, you can make predictions for the answer based on the given context.
            In same cases you have to make decision is good fit candidate ro not you predict he or she is best fit or not.
            you have liberty to make prediction based on context to help user for decision

            Context:
            {similar_docs}

            Question:
            {user_question}

            Answer:
            """
            prompt_parts = [prompt_template]
            model_ge =st.session_state["model_ge"]
            response = model_ge.generate_content(prompt_parts)
            print(response.text)
            st.session_state["chat_history"].append([user_question,response.text])
            st.session_state["prompt"].append(user_question)
            st.session_state["chat_ans_history"].append(response.text)
        st.write(similar_docs)
        if st.session_state["chat_ans_history"]:
            random_key = random.randint(1, 1000000)
            for res,query1 in zip(st.session_state["chat_ans_history"],st.session_state["prompt"]):
                random_key = random.randint(1, 1000000)
                print(res)
                print(query1)
                message(message=query1,is_user=True,key=random_key)
                message(message=res,key=random_key*2)



def Job_Description_evaluation():
    text =''
    st.title("🚀Job Description Recommendations and Enhancements")
    job_description_up=''
    job_review=''
    job_title = st.text_input("Enter the job title")
    job_description = st.text_area("Enter the job description here", height=300)
    flag=0
    if st.button("Craft Stellar Job Descriptions 🌟"):
        Resume_Flag=True
        flag=0
        prompt = f"""Suggest the changes that need to be made for the following job title{job_title} and job description{job_description}:

        Refer to the following example for the output:

        Few Shot Example:

        Job title : Java Developer

        Job Description:

        5+ years of relevant experience in the Financial Service industry
        Intermediate level experience in Applications Development role
        Consistently demonstrates clear and concise written and verbal communication
        Demonstrated problem-solving and decision-making skills
        Ability to work under pressure and manage deadlines or unexpected changes in expectations or requirements

        Output:

        - Add experience requirements to make the role more specific.
        - Include additional skill sets that are essential for the job.
        - Specify the name of the company to personalize the job description.
        - Highlight the unique selling points of the company.
        - Ensure the language is clear, concise, and action-oriented.
        - Emphasize the benefits and perks offered by the company to attract top talent.

        Please provide your suggestions :
        """
        prompt_parts = [prompt]
        model_ge =st.session_state["model_ge"]
        response = model_ge.generate_content(prompt_parts)
        job_review=response.text
        st.title("Suggested Changes")
        st.write(job_review)
        text = "Suggested Changes"+'\n\n'+job_review
        prompt = f"""You have provided a job description and job title for review. Analyze the provided job description based on the job title and suggest potential enhancements to improve its effectiveness. The enhancements will focus on making the job description more attractive and compelling to potential candidates.

        Modify the only given job description. Don't add any information that is not available in the job description.

        Job Title: {job_title}

        Job Description: {job_description}

        Output:
            Enhanced Job Description:
        """
        prompt_parts = [prompt]
        model_ge =st.session_state["model_ge"]
        response = model_ge.generate_content(prompt_parts)
        st.write(response.text)
        job_description_up=response.text
        job_match = re.search(r'Job Title:(.*?)(?=\n\w+:|$)(.*)', job_description_up, re.DOTALL)
        if job_match:
            job_title = job_match.group(1).strip()
            job_description = job_match.group(2).strip()
            print("Job Title:", job_title)
            print("Job Description:", job_description)
        else:
            print("Job Title and/or Job Description not found")
        st.title("Enhanced Job Description")
        st.write(job_description_up)
        text =text+'Enhanced Job Description'+job_description_up
        with open('output.txt', "w") as file:
            file.write(job_description_up)
        with open('output_org.txt', "w") as file:
            file.write(job_description)
        st.success('"Enhanced Job Description...!!!"')




with st.sidebar:
    image = Image.open("HR.png") 
    # Display the image in the sidebar
    st.sidebar.image(image, use_column_width='auto')
    st.sidebar.title("GenAI HR Wizard")
    loaded_model =st.session_state['loaded_model']
    if loaded_model:
        selected_option = option_menu("Talent Evaluation Suite",
            ['Job Description evaluation',"CV Ranking, Generate Screening Questions & Email Send",'First-Round Interview & Evaluation','GenAI Resume Chatbot',
    "Resume Score & Enhancements"]       
             ,icons=['gear', 'sort-numeric-up',  'cloud-upload', 'robot', 'star'],
            menu_icon='file',
            default_index=0,


        )
    preprocessing()
    reset = st.sidebar.button('Reset all')
    if reset:
        st.session_state = {}
        uploaded_file = {}
        




if selected_option=="CV Ranking, Generate Screening Questions & Email Send":
    print("In Cv ranking")
    df_data=[]
    if 'output_data' not in st.session_state:
        st.session_state.output_data = []
    st.title("🔝 Top CV Shortlisting & Ranking, Generate Screening The Questions and Sent The Mail")
    with st.form("resume_short"):
        selected_option_jd = st.selectbox("Select an option:", ["Custom Job Description","Original Job Description", "Enhanced Job Description"])
        if selected_option_jd=="Enhanced Job Description":
            with open('output.txt', 'r') as file:
                content = file.read()
            print("in side en",content)
            job_description = content
            job_description = st.text_area(label="Enhanced Job Description",value=content,height=400)
        elif selected_option_jd=="Original Job Description":
        # Right column for entering job description text
            st.header("Job Description")
            with open('output_org.txt', 'r') as file:
                content = file.read()
            print("in side en",content)
            job_description = content
            job_description = st.text_area(label="Enhanced Job Description",value=content,height=400)
        elif selected_option_jd=="Custom Job Description":
            job_description = st.text_area(label="Enter Job Description",height=400)
        candidate_n = st.number_input("Enter the number of candidates you want to select from the top CV rankings:",min_value=1,step=1)
        l2=[]
        ques=[]
        temp=0
        rank_can =1
        submit_button = st.form_submit_button("CV Ranking 🚀")
        if submit_button and job_description:
            vectorstore_jd = st.session_state["vectorstore_jd"]
            se = vectorstore_jd.similarity_search(job_description,candidate_n)
            st.header("Resume Information According to Rank")
            for i in se:
                st.header("Candidate Information")
                print("----------------------------------------------------------------------------------------")
                #print("Source-------------------",i.metadata['source'].split("\\")[-1])
                #source.append(i.metadata['source'].split("\\")[-1])
                prompt_template = f"""Extract the following Information from the given resume:

                Resume Content:
                {i.page_content}

                Output:
                Name: (e.g., John Doe)
                Job Profile: (e.g., Software Engineer, Data Scientist, etc.)
                Skill Set: (e.g., Python, Machine Learning, SQL, etc.)
                Email: (e.g., john.doe@example.com)
                Phone Number: (e.g., +1 (555) 123-4567)
                Number of Years of Experience: (e.g., 5 years)
                Previous Organizations and Technologies Worked With: (e.g., XYZ Corp - 2 years - Java, ABC Inc - 3 years - Python)
                Education: (e.g., Bachelor of Science in Computer Science, Master of Business Administration, etc.)
                Certifications: (e.g., AWS Certified Developer, Google Analytics Certified, etc.)
                Projects: (e.g., Project Title - Description, Project Title - Description, etc.)
                Location: (e.g., New York, NY, USA)
                """

                prompt_parts = [prompt_template]
                model_ge =st.session_state["model_ge"]
                response = model_ge.generate_content(prompt_parts)
                #dict_info = extract_resume_info(response.text)
                st.write(response.text)
                st.write("\n\n") 

                st.title("🕵️‍♂️Screening Questions")
                prompt_question  = f'''Generate a diverse set of interview questions, including both Five HR and Fifteen Technical questions, tailored to the provided resume and job description:

                Resume:
                {i.page_content}

                Job Description:
                {job_description}

                Please generate a mix of HR and Technical questions that align with the candidate's qualifications and experience, focusing on the following aspects:

                1. Skills: Craft questions that explore the candidate's skills, .
                2. Experience: Generate questions related to the candidate's experience .
                3. Projects: Include inquiries about the candidate's involvement in specific_project mentioned in the resume.
                4. Job Description Alignment: Ensure questions assess the candidate's compatibility with the job_role.
    '''
                prompt_parts = [prompt_question]
                model_ge =st.session_state["model_ge"]
                response_question = model_ge.generate_content(prompt_parts)
                #dict_info = extract_resume_info(response.text)
                st.write(response_question.text)
                df_data.append([response.text,response_question.text])
    df = pd.DataFrame(df_data,columns=["Candidate_Information","Screening_Question"])
    if submit_button:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"resuem_rank_data_{current_time}.csv"
        df.to_csv(file_name, index=False)
        csv_data = BytesIO()
        df.to_csv(csv_data, index=False)
        st.download_button(label="Download Resume Rank CSV File", data=csv_data, file_name=file_name, mime="text/csv")


elif selected_option=='Job Description evaluation':
    print("Function called")
    Job_Description_evaluation()
elif selected_option=='First-Round Interview & Evaluation':
    eval()
elif selected_option=='GenAI Resume Chatbot':
    rss()
elif selected_option=="Resume Score & Enhancements":
    with st.form("resume_score"):
        with st.spinner("Evaluating"):
            st.title("PrecisionScore: Elevating Resumes Through Comprehensive Evaluation ✨")
            name =[]
            uploaded_file = st.file_uploader("Upload your resume:", type=[".pdf"],accept_multiple_files=False)
            btn=st.form_submit_button("Check Score 🌟")
            if uploaded_file and btn:
                name.append(uploaded_file.name)
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file.write(uploaded_file.read())
                temp_file.close()
                resume_score_docs = extract_text_from_pdf(temp_file.name,uploaded_file.name)
                resume_text = resume_score_docs.page_content
                st.header("Resume Score - 🌟")
                prompt_template = f"""Evaluate the following resume and provide a score out of 100 based on the following criteria:

                - **Content:** Evaluate the relevance, accuracy, and completeness of the information provided. Suggest adding specific details to highlight achievements and responsibilities.
                - **Format:** Review the organization, layout, and visual appeal of the resume. Consider using consistent formatting and bullet points for clarity.
                - **Sections:** Check for essential sections such as education, work experience, skills, and certifications. Recommend adding any missing sections that enhance the candidate's profile.
                - **Skills:** Assess the alignment of the candidate's skills with the job requirements. Recommend emphasizing key skills that match the role.
                - **Style:** Evaluate the use of clear and concise language, appropriate tone, and professional writing style. Suggest revising sentences for clarity and impact.

                After scoring, provide constructive feedback to help the candidate improve their resume. Please carefully review the resume and assign a score based on these criteria:

                Example:
                Name: candidate_name
                Score: score
                Positive Feedback: positive_feedback
                Negative Feedback: negative_feedback 
                {resume_text}
                """
                prompt_parts = [prompt_template]
                model_ge =st.session_state["model_ge"]
                response = model_ge.generate_content(prompt_parts)
                dict_info = extract_resume_info(response.text)
                st.write(response.text)
