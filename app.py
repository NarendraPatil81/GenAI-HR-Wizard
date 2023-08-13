
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio
import speech_recognition as sr
from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
import openai
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import os
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.chains import AnalyzeDocumentChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import time
import re
import pandas as pd
from datetime import datetime
from io import BytesIO
import pyperclip
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.question_answering import load_qa_chain
import whisper
from streamlit_option_menu import option_menu


st.set_page_config(page_title="HR Wizard")



# Hide Streamlit footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)



#"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"
#os.environ['OPENAI_API_KEY'] = api_key = os.environ.get('OPENAI_API_KEY')
openai.api_key=api_key = os.environ.get('OPENAI_API_KEY')
from streamlit_chat import message
global docs
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from PIL import Image
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText       
import smtplib
import tempfile
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.document_loaders import UnstructuredFileLoader
resumescore = st.container()
cvranking = st.container()
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=100,
        chunk_overlap=10,
        length_function=len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_documents(text_chunks, embedding=embeddings)
    return vectorstore



def eval():
    st.title("🔍 Decode Interview Performance")
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    # Allow multiple file uploads
    uploaded_files = st.file_uploader("Upload MP4 Files", type=["mp4"], accept_multiple_files=True)
    if uploaded_files and st.button("Evaluate"):
        st.header("Question and Answer with Score")
        #st.write("Uploaded Files:")
        #st.write("Uploaded Files:")
        for resume in uploaded_files:
            #st.write(resume)
            #st.write(resume.name)
            print(resume)
            print(resume.name)
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(resume.read())
            temp_file.close()
            video_path = "uploaded_video.mp4"
            # with open(video_path, "wb") as f:
            #     f.write(temp_file.read())
            model = whisper.load_model("base")
            transcript = model.transcribe(temp_file.name)
            transcript=transcript['text']
            print(transcript)
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            chain = load_qa_chain(llm=OpenAI(), chain_type="map_reduce")
            python_docs = text_splitter.create_documents([transcript])
            que_ans =''
            st_1 =''
            before_overall_feedback_l = []
            after_overall_feedback_l=[]
            for i in python_docs:
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
                completions = openai.Completion.create (engine="text-davinci-003",prompt=prompt_load,max_tokens=2200,n=1,stop=None,temperature=0.5,)
                message = completions.choices[0].text
                #print(message)
                st_1 += message
                before_overall_feedback = re.search(r"(.+?)\nOverall Feedback:", message, re.DOTALL)
                after_overall_feedback = re.search(r"(?<=Overall Feedback:\n)(.+)", message, re.DOTALL)

                if before_overall_feedback:
                    extracted_text_before = before_overall_feedback.group(1)
                    before_overall_feedback_l.append(extracted_text_before)
                    print("Extracted Text Before Overall Feedback:\n", extracted_text_before)
                else:
                    print("Overall Feedback section not found.")

                if after_overall_feedback:
                    extracted_text_after = after_overall_feedback.group(1)
                    after_overall_feedback_l.append(extracted_text_after)
                    print("Extracted Text After Overall Feedback:\n", extracted_text_after)
                else:
                    print("Overall Feedback section not found.")
                # before_overall_feedback = re.search(r"(.+?)\nOverall Feedback:", message, re.DOTALL)
                # if before_overall_feedback:
                #     extracted_text = before_overall_feedback.group(1)
                #     before_overall_feedback_l.append(extracted_text)
                #     st.write(extracted_text)
                #     print("Extracted Text Before Overall Feedback:\n", extracted_text)
                # else:
                #     print("Overall Feedback section not found.")
                # overall_feedback_section = re.search(r"(?<=Overall Feedback:\n)(.+)", message, re.DOTALL)
                # overall.append(overall_feedback_section[0])
                # st.write(overall_feedback_section[0])
                # #print(overall_feedback_section[0])
                for text in before_overall_feedback_l:
                    st.text(text)

            # Display the extracted text after "Overall Feedback"
                st.header("Overall Feedback")
                for text in after_overall_feedback_l:
                    st.text(text)
                text_feed =''
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


def CV_ranking():
    st.title("🔝 Top CV Shortlisting & Ranking, Generate Screening The Questions and Sent The Mail")
    #left_column, right_column = st.columns(2)
    # Left column for uploading multiple PDF resume files
    st.header("Upload Resumes")
    uploaded_files = st.file_uploader("Upload Resumes", type=["txt", "pdf", "docx", "pptx", "html"],accept_multiple_files=True)

    selected_option_jd = st.selectbox("Select an option:", ["Original Job Description", "Enhanced Job Description"])
    if selected_option_jd=="Enhanced Job Description":
        with open('output.txt', 'r') as file:
            content = file.read()
        print("in side en",content)
        job_description = content
        job_description = st.text_area(label="Enhanced Job Description",value=content,height=400)
    elif selected_option_jd=="Original Job Description":
    # Right column for entering job description text
        st.header("Job Description")
        job_description = st.text_area("Enter the job description here..", height=300)
    candidate_n = st.number_input("Enter the number of candidates you want to select from the top CV rankings:",min_value=1,step=1)
    # st.header("Upload Resumes")
    # uploaded_files = st.file_uploader("Upload PDF Resumes", type=["pdf"], accept_multiple_files=True)
    l2=[]
    source=[]
    temp=0
    # If resume files are uploaded, process them
    submit_button = st.button("CV Ranking 🚀")
    if uploaded_files and job_description and submit_button:
        for resume in uploaded_files:
            st.write(resume)
            path = os.path.dirname(__file__)
            st.write(path)
            file_extension = os.path.splitext(resume.name)[1]
            #st.write(f"Uploaded file extension: {file_extension}")
            if file_extension=='.pdf':
                loader = PyPDFLoader(resume.name)
            if file_extension=='.docx':
                loader = UnstructuredWordDocumentLoader(resume.name)
            if file_extension=='.txt':
                loader = UnstructuredFileLoader(resume.name)
            if file_extension=='.pptx':
                loader = UnstructuredPowerPointLoader(resume.name)
            if temp==0:
                temp=1
                docs=loader.load()
                print('docs created')
            else:
                docs=docs+(loader.load())
                print('docs created')
            print(resume)
        embeddings = OpenAIEmbeddings()
        print("uploaded_files--",len(uploaded_files))
        kb = FAISS.from_documents(docs,embeddings)
        se = kb.similarity_search(job_description,candidate_n)
        st.header("Resume Information According to Rank")
        for i in se:
            print("----------------------------------------------------------------------------------------")
            #print("Source-------------------",i.metadata['source'].split("\\")[-1])
            source.append(i.metadata['source'].split("\\")[-1])
            prompt = f"""Extract the following Information from the given resume:

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


            completions = openai.Completion.create (engine="text-davinci-003",prompt=prompt,max_tokens=1500,n=1,stop=None,temperature=0.8)
            message = completions.choices[0].text
            print(message)

            print("Source-------------------",i.metadata['source'].split("\\")[-1])
            resume_info_list = extract_resume_info(message)
            formatted_text = "\n".join([f"{key}: {value}" for key, value in resume_info_list.items()])

            # Display the formatted text
            st.text(formatted_text)
            #st.write(resume_info_list)
            st.text(i.metadata['source'].split("\\")[-1])
            st.write("\n\n")
            l2.append(resume_info_list)
            time.sleep(10)
            st.title("🕵️‍♂️Screening Questions")
            prompt  = f'''Generate a diverse set of interview questions, including both Five HR and Fifteen Technical questions, tailored to the provided resume and job description:

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
            # prompt=f'''Generate a set of Five HR and Fiteen Technical interview questions tailored to the provided resume{resume_text}:
            # Please generate a mix of HR and Technical questions based on the candidate's qualifications and experience.
            # Please generate questions that evaluate both the candidate's interpersonal and technical skills based on their resume.
            # '''  
            completions = openai.Completion.create (engine="text-davinci-003",prompt=prompt,max_tokens=2000,n=1,stop=None,temperature=0.8,)
            questions = completions.choices[0].text
            # questions= chain.run(input_documents=resume_text, question=prompt)
            st.text(questions)
            f_s =str((i.metadata['source'].split("\\")[-1]).split(".")[0])
            print(f_s)
            st.text('Question Saved In '+f_s+'.txt')
            with open(f_s+'.txt', 'w',encoding="utf-8") as f:
                f.write(questions)
            
        print(len(source))
        print(source)
        df = pd.DataFrame(l2)
        #st.write(df)
        print(len(source))
        df['Source']=source
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"resuem_rank_data_{current_time}.csv"
        df.to_csv(file_name, index=False)
        csv_data = BytesIO()
        df.to_csv(csv_data, index=False)
        # Create a download button to download the CSV file
        st.download_button(label="Download Resume Rank CSV File", data=csv_data, file_name=file_name, mime="text/csv")
        st.write(df)
        st.header("Sent Email to Shortlisted Candidates")
        send_email(df)
        
            # After processing, show the job description and results
            #right_column.write(f"### Job Description")
            #right_column.write(job_description)
            # Perform your analysis on the job description and resumes here


def rss():
    #st.set_page_config(page_title="Chat with multiple PDFs",page_icon=":books:")
    st.header("🔍 GenAI-Powered Resume Search Chatbot: Finding the Perfect Fit")
    pdf_docs = st.file_uploader("Upload Resumes", type=["txt", "pdf", "docx", "pptx", "html"],accept_multiple_files=True)
    #prompt1=st.session_state["prompt1"]
    temp=0
    if st.spinner("Processing the files"):
        if not pdf_docs:
            st.stop()
        for uploaded_file in pdf_docs:
            if uploaded_file.name[-4:]=='.pdf':
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file.write(uploaded_file.read())
                temp_file.close()
                loader = PyPDFLoader(temp_file.name)
                #docs=loader.load()
                #st.write(docs)
            if uploaded_file.name[-5:]=='.docx':
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file.write(uploaded_file.read())
                temp_file.close()
                loader = UnstructuredWordDocumentLoader(temp_file.name)
            if uploaded_file.name[-4:]=='.txt':
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file.write(uploaded_file.read())
                temp_file.close()
                loader = UnstructuredFileLoader(temp_file.name)
            if uploaded_file.name[-5:]=='.pptx':
                st.write("in ppt")
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file.write(uploaded_file.read())
                temp_file.close()
                loader = UnstructuredPowerPointLoader(temp_file.name)
            #docs=loader.load()
            #st.write(docs)
            if temp==0:
                temp=1
                docs=loader.load()
                print('docs created')
            else:
                docs=docs+(loader.load())
                print('docs created')
        text_chunks = get_text_chunks(docs)
        vectorstore = get_vectorstore(text_chunks)
        user_question = st.text_input("What type of information are you looking for in these resumes? Enter keywords or skills.")
        if "prompt" not in st.session_state:
            st.session_state.prompt = []
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "chat_ans_history" not in st.session_state:
            st.session_state.chat_ans_history = []
        if user_question:
            print("In side user",user_question)
            #st.session_state['prompt'].append(user_question)
            with st.spinner("Processing"):
                # get pdf text
                chat_history=st.session_state["chat_history"]
                memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
                llm = OpenAI()
                conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,retriever=vectorstore.as_retriever(),memory=memory,verbose=False)
                response=conversation_chain({"question":user_question,"chat_history":chat_history})
                st.session_state["chat_history"].append([user_question,response['answer']])
                st.session_state["prompt"].append(user_question)
                st.session_state["chat_ans_history"].append(response['answer'])
                #print(user_question)
            if st.session_state["chat_ans_history"]:
                for res,query1 in zip(st.session_state["chat_ans_history"],st.session_state["prompt"]):
                    print(res)
                    print(query1)
                    message(query1,is_user=True)
                    message(res)


def send_email(data):
    #name  = data['Name']
    subject = "Next Steps in Hiring Process"
    message = (
    "Congratulations! 🎉 You have been shortlisted for the next steps in the hiring process. "
    "We will be contacting you soon for the screening round."
)
    #message = "Congratulations! 🎉 You have been shortlisted for the next steps in the hiring process. We will be contacting you soon for the screening round."
    for _, row in data.iterrows():
        # Email configuration
        sender_email = "narenj8329@gmail.com"
        receiver_email = row['Email']
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = row['Email']
        msg['Subject'] = "Shortlisted for Next Hiring Round"
        #msg.attach(MIMEText(message, 'plain'))
        msg = MIMEMultipart()
        html_content = f"""
        <html>
        <head></head>
        <body>
            <p>Dear {row['Name']},</p>
            <p>{message}</p>
            <p>Best Regards,<br>Gen AI Wizard</p>
        </body>
        </html>
"""
        msg['Subject'] = 'Congratulations! You have been shortlisted'
        msg.attach(MIMEText(html_content, 'html'))
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, 'ikmvtwehrfeyzibg')
            server.sendmail(sender_email, row['Email'], msg.as_string())
            print(f"Email sent to {row['Email']}")
        except Exception as e:
            print("Error sending email:", str(e))
        finally:
            server.quit()
        res = "Email Sent to "+str(row['Name'])+" at mail "+str(row['Email'])
        st.text(res)



def Job_Description_evaluation():
    text =''
    st.title("🚀Job Description Recommendations and Enhancements")
    #left_column, right_column = st.columns(2)
    job_description_up=''
    job_review=''
    # Left column for uploading multiple PDF resume files
    job_title = st.text_input("Enter the job title")
    # Right column for entering job description text
    job_description = st.text_area("Enter the job description here", height=300)
    flag=0
    # Job Description input
    #st.header("Enter the Job Description")
    #job_description = st.text_area("Job Description", height=200)

    # Job Title input

    # Calculate Score button
    if st.button("Craft Stellar Job Descriptions 🌟"):
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

        completions = openai.Completion.create (
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=3000,
                n=1,
                stop=None,
                temperature=0.3,
            )
        job_review = completions.choices[0].text
        st.title("Suggested Changes")
        st.text(job_review)
        text = "Suggested Changes"+'\n\n'+job_review
        prompt = f"""You have provided a job description and job title for review. Analyze the provided job description based on the job title and suggest potential enhancements to improve its effectiveness. The enhancements will focus on making the job description more attractive and compelling to potential candidates.

        Modify the only given job description. Don't add any information that is not available in the job description.

        Job Title: {job_title}

        Job Description: {job_description}

        Output:
            Enhanced Job Description:
        """
        completions = openai.Completion.create (
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=3000,
                n=1,
                stop=None,
                temperature=0.2,
            )
        job_description_up = completions.choices[0].text

        job_match = re.search(r'Job Title:(.*?)(?=\n\w+:|$)(.*)', job_description_up, re.DOTALL)
        # Extracted job title and job description
        if job_match:
            job_title = job_match.group(1).strip()
            job_description = job_match.group(2).strip()
            print("Job Title:", job_title)
            print("Job Description:", job_description)
        else:
            print("Job Title and/or Job Description not found")
        st.title("Enhanced Job Description")
        st.text(job_description_up)
        text =text+'Enhanced Job Description'+job_description_up
        with open('output.txt', "w") as file:
            file.write(job_description_up)
        with open('output_org.txt', "w") as file:
            file.write(job_description)
        st.text('"Enhanced Job Description Copied! Paste it using Ctrl+V for immediate use...!!!"')
        pyperclip.copy(job_description_up)
        #CV_ranking(job_description_up)



with st.sidebar:
    image = Image.open("HR.png") 
    # Display the image in the sidebar
    st.sidebar.image(image, use_column_width='auto')
    #st.sidebar.title("GenAI HR Wizard")

    selected_option = option_menu(
        'GenAI HR Wizard',
        ['Job Description evaluation',"CV Ranking, Generate Screening Questions & Email Send",'First-Round Interview & Evaluation','GenAI Resume Chatbot',
"Resume Score & Enhancements"]       
         ,icons=['gear', 'sort-numeric-up',  'cloud-upload', 'robot', 'star'],
        menu_icon='file',
        default_index=0,
        styles = {
    "container": {"padding": "10px", "background-color": "#f4f4f4"},
    "icon": {"color": "black", "font-size": "24px"},
    "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
    "nav-link-selected": {"background-color": "#e80e24"}
}

    )
    #options = ['Job Description evaluation',"CV Ranking, Generate Screening Questions & Email Send",'First-Round Interview & Evaluation','GenAI Resume Chatbot',"Resume Score & Enhancements"]
    #selected_option = st.sidebar.radio("Select an option", options)
    reset = st.sidebar.button('Reset all')
    if reset:
        st.session_state = {}
        uploaded_file = {}
        


#docs=[]

if selected_option=="CV Ranking, Generate Screening Questions & Email Send":
    print("In Cv ranking")
    if 'output_data' not in st.session_state:
        st.session_state.output_data = []
    # if "formatted_text_list" not in st.session_state:
    #     st.session_state.formatted_text_list = []
    # if "questions_list" not in st.session_state:
    #     st.session_state.questions_list = []
    st.title("🔝 Top CV Shortlisting & Ranking, Generate Screening The Questions and Sent The Mail")
    st.header("Upload Resumes")
    uploaded_files = st.file_uploader("Upload Resumes", type=["txt", "pdf", "docx", "pptx"],accept_multiple_files=True)
    selected_option_jd = st.selectbox("Select an option:", ["Original Job Description", "Enhanced Job Description"])
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
    candidate_n = st.number_input("Enter the number of candidates you want to select from the top CV rankings:",min_value=1,step=1)
    l2=[]
    ques=[]
    temp=0
    rank_can =1
    # If resume files are uploaded, process them
    submit_button = st.button("CV Ranking 🚀")
    if submit_button:
        if not uploaded_files:
            st.stop()
        for uploaded_file in uploaded_files:
            if uploaded_file.name[-4:]=='.pdf':
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file.write(uploaded_file.read())
                temp_file.close()
                loader = PyPDFLoader(temp_file.name)
                #docs=loader.load()
                #st.write(docs)
            if uploaded_file.name[-5:]=='.docx':
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file.write(uploaded_file.read())
                temp_file.close()
                loader = UnstructuredWordDocumentLoader(temp_file.name)
            if uploaded_file.name[-4:]=='.txt':
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file.write(uploaded_file.read())
                temp_file.close()
                loader = UnstructuredFileLoader(temp_file.name)
            if uploaded_file.name[-5:]=='.pptx':
                st.write("in ppt")
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file.write(uploaded_file.read())
                temp_file.close()
                loader = UnstructuredPowerPointLoader(temp_file.name)
            #docs=loader.load()
            #st.write(docs)
            if temp==0:
                temp=1
                docs=loader.load()
                print('docs created')
            else:
                docs=docs+(loader.load())
                print('docs created')
            #st.write(docs)
        embeddings = OpenAIEmbeddings()
        print("uploaded_files--",len(uploaded_files))
        kb = FAISS.from_documents(docs,embeddings)
        se = kb.similarity_search(job_description,candidate_n)
        st.header("Resume Information According to Rank")
        for i in se:
            print("----------------------------------------------------------------------------------------")
            #print("Source-------------------",i.metadata['source'].split("\\")[-1])
            #source.append(i.metadata['source'].split("\\")[-1])
            prompt = f"""Extract the following Information from the given resume:

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


            completions = openai.Completion.create (engine="text-davinci-003",prompt=prompt,max_tokens=1500,n=1,stop=None,temperature=0.8)
            message = completions.choices[0].text
            resume_info_list = extract_resume_info(message)
            can_name = resume_info_list["Name"]
            formatted_text = "\n".join([f"{key}: {value}" for key, value in resume_info_list.items()])
            st.header("Rank "+str(rank_can)+" Candidate Name - "+str(can_name))
            rank_can +=1
            st.text(formatted_text)
            #st.session_state.formatted_text_list.append(formatted_text)
            #st.write(resume_info_list)

            #st.write("\n\n")
            l2.append(resume_info_list)
            time.sleep(10)
            st.title("🕵️‍♂️Screening Questions")
            prompt  = f'''Generate a diverse set of interview questions, including both Five HR and Fifteen Technical questions, tailored to the provided resume and job description:

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
            # prompt=f'''Generate a set of Five HR and Fiteen Technical interview questions tailored to the provided resume{resume_text}:
            # Please generate a mix of HR and Technical questions based on the candidate's qualifications and experience.
            # Please generate questions that evaluate both the candidate's interpersonal and technical skills based on their resume.
            # '''  
            completions = openai.Completion.create (engine="text-davinci-003",prompt=prompt,max_tokens=2000,n=1,stop=None,temperature=0.8,)
            questions = completions.choices[0].text
            ques.append(questions)
            # questions= chain.run(input_documents=resume_text, question=prompt)
            st.text(questions)
            st.session_state.output_data.append({
        'nested_list': formatted_text,
        'string_list': questions
    })
            #st.session_state.questions_list.append(questions)

            #st.text('Question Saved In '+str(can_name)+'.txt')
            # with open(str(can_name)+'.txt', 'w',encoding="utf-8") as f:
            #     f.write(questions)
            
            #st.download_button(label="Download the Question for the candidate", data=questions, file_name=((can_name)+'.txt'), mime="text/plain")
        #df = pd.DataFrame(l2)
        #st.write(df)
        data = []
        for candidate_info, questions in zip(l2, ques):
            row = [candidate_info[key] for key in candidate_info]
            row.append(questions)
            data.append(row)

        columns = list(l2[0].keys()) + ['Questions']
        df = pd.DataFrame(data, columns=columns)
        st.dataframe(df)
        # resume_info = st.session_state.formatted_text_list
        # question_list = st.session_state.questions_list
        # st.write("de")
        # st.text(resume_info)
        # st.text(question_list)
        # if resume_info is None or question_list is None:
        #     st.write("Please upload resume information and questions.")

        # else:
        #     # Display resume information and questions for each candidate
        #     idx=1
        #     for candidate_info, questions in zip(resume_info, question_list):
        #         st.header("Candidate Rank - ",str(idx)," Name - ",candidate_info["Name"])
        #         st.subheader("Candidate Info")
        #         st.text(candidate_info)
        #         print(candidate_info)
        #         st.subheader("Questions:")
        #         st.text(questions)
        #         print(questions)
        #         idx +=1

        #         st.write("-" * 30)  # Separator





        # Use a separate button to trigger download
        # if st.button("Download DataFrame with Rank and Questions"):
        #     csv_data = df.to_csv(index=False)
        #     file_name = "resume_data_rank_questions.csv"
        #     st.download_button(label="Download Resume Data with Rank and Questions as CSV File", data=csv_data, file_name=file_name, mime="text/csv")
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"resuem_rank_data_{current_time}.csv"
        df.to_csv(file_name, index=False)
        csv_data = BytesIO()
        df.to_csv(csv_data, index=False)
        st.download_button(label="Download Resume Rank CSV File", data=csv_data, file_name=file_name, mime="text/csv")
        st.session_state['data_file'] = df
        # Create a download button to download the CSV file
    # if st.session_state.output_data:
    #     for idx, output in enumerate(st.session_state.output_data):
    #         st.write(f"Output Set {idx + 1}")
    #         st.write("Nested List with Dict:")
    #         st.write(output['nested_list'])
    #         st.write("List with Strings:")
    #         st.write(output['string_list'])
    #         st.write("-" * 20)
        
    
        # st.write(df)
        # st.header("Sent Email to Shortlisted Candidates")
#         send_email(df)
        
            # After processing, show the job description and results
            #right_column.write(f"### Job Description")
            #right_column.write(job_description)
            # Perform your analysis on the job description and resumes here


elif selected_option=='Job Description evaluation':
    print("Function called")
    Job_Description_evaluation()
elif selected_option=='First-Round Interview & Evaluation':
    eval()
elif selected_option=='Generate Screening Questions':
    gsq()
elif selected_option=='GenAI Resume Chatbot':
    rss()
elif selected_option=="Resume Score & Enhancements":
    with resumescore:
        temp=0
        st.title("PrecisionScore: Elevating Resumes Through Comprehensive Evaluation ✨")
        name =[]
        uploaded_files = st.file_uploader("Upload your resume:", type=[".pdf", ".docx",".csv",".pptx"],accept_multiple_files=True)
        if not uploaded_files:
            st.stop()              
        if uploaded_files and st.button("Check Score"):
            for uploaded_file in uploaded_files:
                name.append(uploaded_file.name)
                if uploaded_file.name[-4:]=='.pdf':
                    temp_file = tempfile.NamedTemporaryFile(delete=False)
                    temp_file.write(uploaded_file.read())
                    temp_file.close()
                    loader = PyPDFLoader(temp_file.name)
                    #docs=loader.load()
                    #st.write(docs)
                if uploaded_file.name[-5:]=='.docx':
                    temp_file = tempfile.NamedTemporaryFile(delete=False)
                    temp_file.write(uploaded_file.read())
                    temp_file.close()
                    loader = UnstructuredWordDocumentLoader(temp_file.name)
                if uploaded_file.name[-4:]=='.txt':
                    temp_file = tempfile.NamedTemporaryFile(delete=False)
                    temp_file.write(uploaded_file.read())
                    temp_file.close()
                    loader = UnstructuredFileLoader(temp_file.name)
                if uploaded_file.name[-5:]=='.pptx':
                    st.write("in ppt")
                    temp_file = tempfile.NamedTemporaryFile(delete=False)
                    temp_file.write(uploaded_file.read())
                    temp_file.close()
                    loader = UnstructuredPowerPointLoader(temp_file.name)
                #docs=loader.load()
                #st.write(docs)
                if temp==0:
                    temp=1
                    docs=loader.load()
                    print('docs created')
                else:
                    docs=docs+(loader.load())
                    print('docs created')
                #st.write(docs)
            #st.write(name)
            #st.write(uploaded_files)
            for docs_list,name_res in zip(docs,name):
                #st.write(docs_list)
                resume_text = docs_list.page_content
                st.header("Resume Score -")
                prompt = f"""Evaluate the following resume and provide a score out of 100 based on the following criteria:

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
                completions = openai.Completion.create (engine="text-davinci-003",prompt=prompt,max_tokens=2200,n=1,stop=None,temperature=0.5,)
                message1 = completions.choices[0].text
                items = message1.split(".")
                for i in items:
                    st.text(i)







