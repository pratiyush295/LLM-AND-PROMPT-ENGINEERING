from flask import Flask, render_template, request, jsonify
import openai
from openai import OpenAI
import PyPDF2
import os
import json
from openai_schema.resume_schema import ResumeSchema
from openai_schema.linkedin_schema import LinkedinResumeSchema
import urwid

app = Flask(__name__)

# Route for the home page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/generate_resume', methods=['POST'])
def generate_resume():
    
    api_key = request.form['api_key']
    pdf_file = request.files['pdf_file']

    # Set OpenAI API key
    # openai.api_key = api_key

    openai=OpenAI(api_key=api_key)
    
    
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()

    # Send the extracted PDF text to OpenAI for generating an HTML resume
    response = openai.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that transforms resumes into HTML format."
            },
                {
                    "role": "user",
                    "content": f"Convert the following LinkedIn resume into an HTML resume:\n\n{pdf_text}"
                }
            ],
        response_format = LinkedinResumeSchema
        )

        # Extract the generated HTML content from the response
    resume_response = response.choices[0].message.parsed
    resume_response = resume_response.model_dump_json()
    parsed_resume = json.loads(resume_response) 
    # print(html_resume.contact_info)
    return render_template('linkedin.html', **parsed_resume)

if __name__ == '__main__':
    app.run(debug = False, host = "0.0.0.0")
