#Base Image to use
FROM amd64/python:3.10-slim


RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip
RUN pip install contentful rich-text-renderer beautifulsoup4 
RUN pip install streamlit streamlit-chat 
RUN pip install farm-haystack 
RUN pip install 'farm-haystack[faiss]'
RUN pip install tiktoken


#Expose port 8080
EXPOSE 8080

#Copy Requirements.txt file into app directory
COPY requirements.txt app/requirements.txt
	
#install all requirements in requirements.txt
RUN pip install -r app/requirements.txt

#Copy all files in current directory into app directory
COPY . /app

#Change Working Directory to app directory
WORKDIR /app

#Run the application on port 8080
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
