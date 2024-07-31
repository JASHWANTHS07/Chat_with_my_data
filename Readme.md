# Chat with My Data

"Chat with My Data" is an interactive application that allows users to upload a PDF file and ask questions based on the content of that file. The project leverages Natural Language Processing (NLP) techniques to analyze the content and provide insightful responses. It includes various evaluation metrics to ensure the quality, relevance, and coherence of the responses.

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Contributing](#contributing)
6. [License](#license)

## Features

- Upload a PDF file and extract its content.
- Ask questions based on the content of the uploaded PDF.
- Evaluate the responses using various metrics.
- Ensure the responses are safe, unbiased, and non-toxic.
- Provide comprehensive evaluation scores for the responses.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/JASHWANTHS07/Chat_with_my_data.git
    cd Chat_with_my_data.git
    ```

2. **Create a virtual environment:**

    ```bash
    python -m venv venv
    ```

3. **Activate the virtual environment:**

    - On Windows:

        ```bash
        venv\Scripts\activate
        ```

    - On macOS/Linux:

        ```bash
        source venv/bin/activate
        ```

4. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the application:**

    ```bash
    streamlit run [your_path]/app2.py
    ```

2. **Upload a PDF file:**
   - Use the file uploader to select and upload a PDF file.

3. **Ask a question:**
   - Enter your question in the provided input field.

4. **View the response:**
   - The application will display the response generated based on the PDF content.

5. **Evaluate the response:**
   - The application will show various metrics to evaluate the response.

## Project Structure

```
ChatwithMyData/
├── app.py                    # Main application script
├── eval_metrics.py            # Script containing evaluation metrics and LLMasEvaluator class
├── requirements.txt           # List of required packages
├── README.md                  # Project documentation
├── .venv/                     # Virtual environment (not included in the repository)
└── data/                      # Directory to store uploaded PDF files
```

### app.py

The main Streamlit application script that handles:
- PDF file upload.
- Question input.
- Response generation.
- Displaying evaluation metrics.

### eval_metrics.py

Contains:
- `LLMasEvaluator`: A class for evaluating responses.
- Various utility functions and decorators for evaluation.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to customize the content according to your project specifics. This README provides a clear overview of the project, its features, and how to get started.
