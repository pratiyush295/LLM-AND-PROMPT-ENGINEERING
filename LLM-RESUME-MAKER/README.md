# LinkedIn Resume to HTML Converter
link - https://linkedin-resume-maker.vercel.app/

This project is a Flask-based web application that allows users to upload a LinkedIn profile PDF and convert it into a structured, visually-appealing HTML resume using OpenAI's GPT models. The application takes advantage of natural language processing (NLP) to transform PDF content into a JSON response, which is then rendered as HTML using Jinja2 templates.

## Key Features
- **LinkedIn PDF Upload**: Allows users to upload their LinkedIn profile in PDF format for processing.
- **Resume Conversion to HTML**: Utilizes OpenAI's GPT models to intelligently convert the raw content into a well-structured resume in HTML format.
- **Jinja2 Templating**: Dynamically renders the converted resume using Jinja2, customizing the display according to the extracted data schema.
- **Scalable API Design**: Supports easy deployment on cloud platforms like Vercel for high scalability.

## Tech Stack Overview
- **Backend Framework**: Flask (lightweight and efficient for small projects)
- **API**: OpenAI GPT-4 model, utilized for text-to-HTML transformation
- **PDF Parsing**: PyPDF2, used to extract text from uploaded PDFs
- **Templating Engine**: Jinja2, for rendering the final HTML structure
- **Deployment**: Vercel, chosen for its seamless integration with Python-based apps

## Detailed Approach

### 1. Choice of Flask Framework
The project uses the Flask framework due to its simplicity and lightweight design, making it a suitable choice for rapid prototyping and handling basic request-response cycles. While other frameworks like FastAPI were considered for their performance benefits in async APIs, Flask was chosen because its synchronous request handling suffices for this application.

### 2. PDF Parsing and Text Extraction
The uploaded LinkedIn profile PDF is processed using the `PyPDF2` library. This library extracts the raw text from the PDF, which is then used to feed the OpenAI API.

#### Process Flow:
- The PDF is read, and each page’s content is extracted and concatenated to form a single text string.
- This extracted content is sent as a message to OpenAI’s GPT model along with a pre-defined prompt.

### 3. OpenAI API for Resume Generation
After extracting the PDF content, the text is sent to OpenAI's GPT-4 model. The prompt guides the model to generate a well-structured resume using a custom schema (`LinkedinResumeSchema`).

- The API uses the GPT-4 model (`gpt-4o-mini-2024-07-18`) to generate the resume in JSON format based on the extracted LinkedIn data.
- A system message defines the task, telling GPT to format the resume content as per a predefined schema (stored in `openai_schema/linkedin_schema.py`).
- The model returns a structured JSON response with various fields (e.g., contact info, experience, skills) filled based on the PDF content.

### 4. Schema Definition for Consistency
The `LinkedinResumeSchema` (in `openai_schema/linkedin_schema.py`) is a Pydantic BaseModel class that defines the expected structure of the resume. This ensures that the response from the OpenAI model adheres to a consistent format, allowing easy conversion into HTML.

#### Example Structure:
- **Contact Information**: Name, email, phone number
- **Experience**: Previous jobs, positions, and responsibilities
- **Skills**: Technical and non-technical skills relevant to the user’s profile

### 5. Jinja2 Templating for HTML Rendering
The JSON response from OpenAI is parsed and injected into a Jinja2 template (`linkedin.html`). This template is designed to format the structured data into a clean and professional-looking HTML resume.

- The template dynamically handles the schema fields and renders them accordingly, ensuring a customizable and flexible layout.
- Any missing data in the schema is handled gracefully, ensuring that the final HTML resume remains functional even if some fields are incomplete.

### 6. Deployment on Vercel
The application is designed for easy deployment on Vercel. Vercel supports Python-based serverless functions, making it an ideal platform for deploying small Flask applications that require scalability and flexibility.

## API Endpoints

### 1. `/` (Home)
- **Method**: `GET`
- **Description**: Renders the home page where users can submit their OpenAI API key and LinkedIn PDF file.
- **Response**: HTML form for submitting API key and PDF.

### 2. `/generate_resume`
- **Method**: `POST`
- **Description**: Processes the uploaded PDF and generates an HTML resume using OpenAI's GPT model.
- **Parameters**:
  - `api_key`: OpenAI API key (submitted via form)
  - `pdf_file`: LinkedIn PDF file
- **Process**:
  - Extracts the text from the PDF using `PyPDF2`.
  - Sends the extracted content to OpenAI API with a pre-defined prompt.
  - Receives the structured response in JSON format based on `LinkedinResumeSchema`.
  - Renders the response into HTML using the Jinja2 template `linkedin.html`.
- **Response**: The generated HTML resume is displayed on the browser.

## Project Structure

. ├── app.py # Main Flask application ├── templates/ │ └── index.html # Form for submitting API key and PDF │ └── linkedin.html # Renders the final HTML resume ├── openai_schema/ │ └── resume_schema.py # Contains schema for resume structure │ └── linkedin_schema.py # Schema for LinkedIn resumes ├── README.md # Project documentation └── requirements.txt # Python dependencies




###How to run:
1. virtualenv env {in your desired folder}
2. env/Script/activate {in bash}
3. pip install -r requirements.txt
4. python app.py
