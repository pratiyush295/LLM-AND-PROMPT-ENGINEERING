
# CSV Merge Programmatically

This README file provides step-by-step instructions to execute the CSV merge process programmatically using the provided Python notebook (`CSV_MERGE_PROGRAMATICALLY.ipynb`). This guide covers environment setup, code execution, and an understanding of the workflow.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Executing the Notebook](#executing-the-notebook)
4. [Understanding the Workflow](#understanding-the-workflow)
5. [Example Usage](#example-usage)
6. [FAQ](#faq)

---

## 1. Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Python 3.x**: Download and install from [Python's official website](https://www.python.org/downloads/).
- **Jupyter Notebook**: Install using pip or through Anaconda.

## 2. Installation

### Step 1: Install Python

Download and install Python from [here](https://www.python.org/downloads/). Ensure the Python path is added to your system’s environment variables.

### Step 2: Set Up a Virtual Environment (Optional but Recommended)

Create a virtual environment to isolate your project dependencies. Run the following commands:

```bash
# Install virtualenv if you don't have it already
pip install virtualenv

# Create a virtual environment
virtualenv myenv

# Activate the virtual environment
# On Windows:
myenv\Scripts\activate
# On Mac/Linux:
source myenv/bin/activate
```

### Step 3: Install Required Python Libraries

Navigate to the directory containing your notebook (`CSV_MERGE_PROGRAMATICALLY.ipynb`) and install the required libraries:

```bash
pip install -r requirements.txt
```

If a `requirements.txt` file is not provided, install the libraries manually:

```bash
pip install pandas openai jupyter
```

## 3. Executing the Notebook

### Step 1: Launch Jupyter Notebook

Run the following command in your terminal or command prompt to launch Jupyter Notebook:

```bash
jupyter notebook
```

This will open Jupyter Notebook in your default web browser.

### Step 2: Open the Notebook

- Navigate to the directory containing `CSV_MERGE_PROGRAMATICALLY.ipynb`.
- Click on the notebook file to open it.

### Step 3: Run the Cells

- Execute each cell sequentially by pressing `Shift + Enter` or by clicking the "Run" button.
- Ensure you run all cells to execute the complete code.

## 4. Understanding the Workflow

This notebook helps you merge multiple CSV files programmatically by:

1. **Analyzing CSV Files**: Identifies similar columns based on column names and data content.
2. **Generating Mapping**: Creates a dictionary mapping similar columns across all files.
3. **Renaming Columns**: Renames columns using the mapping for consistency.
4. **Merging CSV Files**: Merges the files vertically while handling discrepancies like different data formats or missing data.
5. **Output**: Produces a merged CSV file and a mapping dictionary.

## 5. Example Usage

### Sample Code to Use in a Python Script

```python
import openai
import pandas as pd

# Function to generate mapping response
def generate_mapping_response(files):
    response = openai.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": "You are a data processing expert specializing in merging and analyzing financial and accounts data from multiple CSV files."
            },
            {
                "role": "user",
                "content": "Give only the mapping as response, nothing else, response should be a Dictionary string so that I can easily convert it to a dictionary."
            },
            {
                "role": "user",
                "content": f"Here are the contents of the CSV files: {files}"
            }
        ],
        temperature=0,
    )
    return response.choices[0].message.content

# Reading CSV files
file1 = pd.read_csv('file1.csv')
file2 = pd.read_csv('file2.csv')

# Files dictionary to pass to the function
files_dict = {
    'File-1': file1,
    'File-2': file2
}

# Generate the mapping
mapping_response = generate_mapping_response(files_dict)
print(mapping_response)
```

## 6. FAQ

**Q1: What should I do if the notebook doesn't execute properly?**

- Ensure all dependencies are installed.
- Check for any error messages in the terminal or notebook cells, and troubleshoot accordingly.

**Q2: How do I handle missing data in CSV files?**

- The notebook includes handling NaN values when merging data. You can customize this behavior as needed.

**Q3: Can I use this script for files other than financial or accounts data?**

- The script is optimized for financial and accounts data, but you can modify the logic for other data types.
