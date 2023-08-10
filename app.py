
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
os.environ['OPENAI_API_KEY'] = "enter-api-key"
openai.api_key="enter-api-key"
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
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.document_loaders import UnstructuredFileLoader
#image = Image.open("https://github.com/NarendraPatil81/GenAI-HR-Wizard/blob/5c4ffdf60213d7da3dfa1613974a6ed686ef4abb/HR.png")  # Replace with the actual path to your image

# Display the image in the sidebar
#st.sidebar.image(image, use_column_width=200)

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

def save_uploaded_file(file):
    print("file uploaded")
    drive_path = "/content/drive/MyDrive/Upload2"  # Change to your desired Google Drive folder
    os.makedirs(drive_path, exist_ok=True)
    file_path = os.path.join(drive_path, file.name)
    with open(file_path, "wb") as f:
        f.write(file.read())
    return file_path

def save_video_file(file):
    print("file uploaded")
    drive_path = "/content/drive/MyDrive/HR_Video"  # Change to your desired Google Drive folder
    os.makedirs(drive_path, exist_ok=True)
    file_path = os.path.join(drive_path, file.name)
    with open(file_path, "wb") as f:
        f.write(file.read())
    return file_path

def eval():
    st.title("🔍 Decode Interview Performance")
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    # Allow multiple file uploads
    uploaded_files = st.file_uploader("Upload MP4 Files", type=["mp4"], accept_multiple_files=True)
    if uploaded_files:
        st.header("Question and Answer with Score")
        #st.write("Uploaded Files:")
        #st.write("Uploaded Files:")
        for resume in uploaded_files:
            #st.write(resume)
            #st.write(resume.name)
            print(resume)
            print(resume.name)
            
            video_path = "uploaded_video.mp4"
            with open(video_path, "wb") as f:
                f.write(resume.read())
            model = whisper.load_model("base")
            transcript = model.transcribe(video_path)
            transcript=transcript['text']
            print(transcript)
            # Use regular expressions to extract questions, answers, and feedback
            # pattern = r'Question: (.*?)\n\nAnswer: (.*?)\n\nScore: (\d+)'  # Pattern for extracting question, answer, and score
            # qa_matches = re.findall(pattern, text, re.DOTALL)

            # pattern = r'Overall Feedback:\n\n(.*?)\n\n'  # Pattern for extracting overall feedback
            # feedback_matches = re.findall(pattern, text, re.DOTALL)

            # # Organize the extracted data into a list of dictionaries
            # qa_list = [{'question': q, 'answer': a, 'score': int(s)} for q, a, s in qa_matches]
            # feedback_list = [{'feedback': f} for f in feedback_matches]

            # # Combine the questions, answers, scores, and feedback into a final list of dictionaries
            # final_data = []
            # for i in range(len(qa_list)):
            #     qa_dict = qa_list[i]
            #     qa_dict.update(feedback_list[i])
            #     final_data.append(qa_dict)

            # # Print the extracted data
            # for item in final_data:
            #     st.text(item['question'])
            #     st.text(item['answer'])
            #     st.text(item['score'])
            #     st.text(item['feedback'])
            #     print("Question:", item['question'])
            #     print("Answer:", item['answer'])
            #     print("Score:", item['score'])
            #     print("Overall Feedback:", item['feedback'])
            #     print("----")
            # # # Extract audio from the uploaded video
            # audio_path = "audio.wav"
            # ffmpeg_extract_audio(video_path, audio_path)
    
            # st.write("Audio extracted successfully!")
    
            # # Initialize the recognizer
            # recognizer = sr.Recognizer()
    
            # # Load the audio file and perform speech recognition
            # with sr.AudioFile(audio_path) as source:
            #     st.write("Performing speech recognition...")
            #     audio = recognizer.record(source)
    
            #     try:
            #         transcript = recognizer.recognize_google(audio)
            #         st.write("Extracted Text:")
            #         st.write(transcript)
            #     except sr.UnknownValueError:
            #         st.write("Speech Recognition could not understand audio")
            #     except sr.RequestError as e:
            #         st.write(f"Could not request results from Google Web Speech API; {e}")

        # Clean up: Delete the temporary audio file
            #speech = transcript
            #model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
            #model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin" # the model is in bin format
            #model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
            #n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
            #n_batch = 256  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

            # Loading model,
            #llm = LlamaCpp(model_path=model_path,max_tokens=256,n_gpu_layers=n_gpu_layers,n_batch=n_batch,callback_manager=callback_manager,n_ctx=1024,verbose=False,)
            #chain=load_qa_chain(llm, chain_type="stuff")
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
                completions = openai.Completion.create (engine="text-davinci-003",prompt=prompt_load,max_tokens=2200,n=1,stop=None,temperature=0.8,)
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
            
            if st.button("Save Feedback"):
                with open("candidate_evaluation_feedback.txt", "w") as file:
                    for before_text, after_text in zip(before_overall_feedback_l, after_overall_feedback_l):
                        file.write("Question and Answers:\n")
                        file.write(before_text + "\n\n")
                        file.write("'Overall Feedback':\n")
                        file.write(after_text + "\n\n")
                st.text("Feedback Saved to 'candidate_evaluation_feedback.txt'")



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


# def data_loader():
#     file_extension = os.path.splitext(resume.name)[1]
#     if file_extension=='.pdf':
#         loader = PyPDFLoader(resume.name)
#         docs=loader.load()
#     if file_extension=='.docx':
#         loader = UnstructuredWordDocumentLoader(resume.name)
#         docs=loader.load()
#     if file_extension=='.txt':
#         loader = UnstructuredFileLoader(resume.name)
#         docs=loader.load()
#     if file_extension=='.pptx':
#         loader = UnstructuredPowerPointLoader(resume.name)
#         docs=loader.load()


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
            with open(f_s+'.txt', 'w') as f:
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


def CV_ranking_job_des(job_description):
    st.header('CV Ranking')
    # left_column, right_column = st.columns(2)

    # # Left column for uploading multiple PDF resume files
    # left_column.header("Upload Resumes")
    # uploaded_files = left_column.file_uploader("Upload PDF Resumes", type=["pdf"], accept_multiple_files=True)

    # # Right column for entering job description text
    # right_column.header("Job Description")
    # job_description = right_column.text_area("Enter the job description here", height=300)
    st.header("Upload Resumes")
    uploaded_files = st.file_uploader("Upload PDF Resumes", type=["pdf"], accept_multiple_files=True)
    l2=[]
    source=[]
    temp=0
    # If resume files are uploaded, process them
    submit_button = st.button("Submit")
    if uploaded_files and job_description and submit_button:
        for resume in uploaded_files:
                loader = PyPDFLoader(resume)
                if temp==0:
                    temp=1
                    docs=loader.load()
                    print('docs created')
                else:
                    docs=docs+(loader.load())
                    print('docs created')
                print(resume,type(resume))
        embeddings = OpenAIEmbeddings()
        kb = FAISS.from_documents(docs,embeddings)
        se = kb.similarity_search(job_description,len(uploaded_files))
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
            time.sleep(18)
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
            # After processing, show the job description and results
            #right_column.write(f"### Job Description")
            #right_column.write(job_description)
            # Perform your analysis on the job description and resumes here
global job_description_up,job_review
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
        st.text('"Enhanced Job Description Copied! Paste it using Ctrl+V for immediate use...!!!"')
        pyperclip.copy(job_description_up)
        #CV_ranking(job_description_up)



def gsq():
    st.title("🕵️‍♂️Generate Screening Questions")
    # Left column for uploading multiple PDF resume files
    st.header("Upload Resumes")
    uploaded_files = st.file_uploader("Upload PDF Resumes", type=["pdf"], accept_multiple_files=True)

    # Right column for entering job description text
    st.header("Job Description")
    job_description = st.text_area("Enter the job description here", height=300)
    for pdf_file in uploaded_files:
        if pdf_file:
            # video_path = "cv.pdf"
            # with open(video_path, "wb") as f:
            #     f.write(pdf_file.read())
            loader = PyPDFLoader(pdf_file.name)
            resume_text = loader.load()
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            chain = load_qa_chain(llm=OpenAI(), chain_type="map_reduce")
            #python_docs = text_splitter.create_documents([resume_text])
            if st.button("Generate Questions"):
                if resume_text:
                    prompt  = f'''Generate a diverse set of interview questions, including both Five HR and Fifteen Technical questions, tailored to the provided resume and job description:

                    Resume:
                    {resume_text}

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
                    f_s = pdf_file.name
                    f_s = f_s.split(".")[0]
                    print(f_s)
                    st.text('Question Saved In '+f_s+'.txt')
                    with open(f_s+'.txt', 'w') as f:
                        f.write(questions)
                else:
                    st.warning("Unable to extract text from the PDF.")


def rss():
    #st.set_page_config(page_title="Chat with multiple PDFs",page_icon=":books:")
    st.header("🔍 GenAI-Powered Resume Search Chatbot: Finding the Perfect Fit")
    pdf_docs = st.file_uploader("Upload resumes and click on ", accept_multiple_files=True)
    #prompt1=st.session_state["prompt1"]
    temp=0
    if pdf_docs:
        for resume in pdf_docs:
            loader = PyPDFLoader(resume.name)
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

def calculate_resume_score():
    st.title("PrecisionScore: Elevating Resumes Through Comprehensive Evaluation ✨")
    # Input area for the resume text
    uploaded_file = st.file_uploader("Upload your resume:", type=["pdf", "docx"])

    if st.button("Check Score"):
        loader = PyPDFLoader(uploaded_file.name)
        docs=loader.load()
        resume_text = docs[0].page_content
        prompt = f"Evaluate the following resume and provide a score out of 100 based on the following criteria:\n\
        - Content: Evaluate the relevance, accuracy, and completeness of the information provided. Suggest adding specific details to highlight achievements and responsibilities.\n\
        - Format: Review the organization, layout, and visual appeal of the resume. Consider using consistent formatting and bullet points for clarity.\n\
        - Sections: Check for essential sections such as education, work experience, skills, and certifications. Recommend adding any missing sections that enhance the candidate's profile.\n\
        - Skills: Assess the alignment of the candidate's skills with the job requirements. Recommend emphasizing key skills that match the role.\n\
        - Style: Evaluate the use of clear and concise language, appropriate tone, and professional writing style. Suggest revising sentences for clarity and impact.\n\
        After scoring, provide constructive feedback to help the candidate improve their resume. Please carefully review the resume and assign a score based on these criteria:\n\
        {resume_text}"
        completions = openai.Completion.create (engine="text-davinci-003",prompt=prompt,max_tokens=2200,n=1,stop=None,temperature=0.8,)
        message = completions.choices[0].text
        message=message.split(".")
        for i in message:
            st.text(i)
        



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
        sender_email = "enter_your_mail@gmail.com"
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
            server.login(sender_email, 'password')
            server.sendmail(sender_email, row['Email'], msg.as_string())
            print(f"Email sent to {row['Email']}")
        except Exception as e:
            print("Error sending email:", str(e))
        finally:
            server.quit()
        res = "Email Sent to "+str(row['Name'])+" at mail "+str(row['Email'])
        st.text(res)

#uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx", "pptx", "html"])


 
def main():
    docs=[]
    st.sidebar.title("GenAI HR Wizard")
    options = ['Job Description evaluation',"CV Ranking, Generate Screening Questions & Email Send",'First-Round Interview & Evaluation','GenAI Resume Chatbot',
"Resume Score & Enhancements"]
    selected_option = st.sidebar.radio("Select an option", options)

    if selected_option=="CV Ranking, Generate Screening Questions & Email Send":
        print("In Cv ranking")
        CV_ranking()
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
        calculate_resume_score()



# def read_pdf(file):
#     # Function to read the content of a PDF file
#     # You can use libraries like PyPDF2, pdfplumber, or pdfminer to read PDF content
#     # For simplicity, let's assume we're just reading text from the PDF here.
#     if file is not None:
#         return file.read().decode()


if __name__ == "__main__":
    main()

