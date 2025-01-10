# PDF Chat App 📝📚

This project is a **Streamlit-based application** that allows users to upload a PDF document, ask questions about its content, and get context-aware answers. It uses **LangChain**, **Google Generative AI**, and a **Retrieval-Augmented Generation (RAG)** pipeline to achieve highly accurate and contextual responses.

## Features 🚀

- **Upload PDFs:** Users can upload any PDF document.
- **Chat Interface:** Ask questions about the uploaded PDF content in a user-friendly chat interface.
- **RAG Pipeline:** Utilizes LangChain for document splitting, embeddings creation, and retrieval.
- **Contextual Answers:** Provides accurate answers based on the uploaded document.
- **Conversation History:** Saves Q&A sessions with timestamps in a CSV file.
- **Streamlit Integration:** A seamless and intuitive user experience.
- **Google Generative AI Integration:** For generating embeddings and AI-based responses.

## Tech Stack 🛠️

- **Python**: Core programming language.
- **Streamlit**: For creating the web application interface.
- **LangChain**: For the RAG pipeline and LLM integration.
- **Google Generative AI**: For embeddings and chat responses.
- **FAISS**: For vector similarity search and retrieval.

---

## How to Run the Project 🏃‍♂️

### Prerequisites

1. **Install Python 3.8+**
2. Install the required libraries by running:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with your **Google Generative AI credentials**:

   ```env
   GOOGLE_API_KEY=your_google_api_key
   ```

### Steps to Run

1. Clone the repository:

   ```bash
   [git clone https://github.com/your-repo-name/pdf-chat-app.git](https://github.com/prashantsingh5/RAG-based_Chatbot.git)
   cd RAG-based_Chatbot
   ```

2. Start the Streamlit application:

   ```bash
   streamlit run app.py
   ```

3. Open your browser and navigate to `http://localhost:8501` to interact with the app.

---

## File Structure 📂

```
pdf-chat-app/
├── proj4.py                 # Main application file
├── requirements.txt       # Python dependencies
├── record.csv             # Logs conversations with timestamps (generated after interaction)
└── README.md              # Project documentation
```

---

## Usage Instructions 🖐️

1. Upload a PDF document using the "Choose a PDF file" option.
2. Once the PDF is processed, type your question into the text input field.
3. View the AI-generated answer in real time.
4. Check the "Conversation History" for past interactions.

---

## Example Q&A Session 💬

| **Timestamp**           | **PDF Name**    | **Question**            | **Answer**             |
|--------------------------|-----------------|-------------------------|------------------------|
| 10-01-2025  12:37:49     | yolov9_paper.pdf  | what is Auxiliary Supervision? | Auxiliary supervision is a common method that uses relevant meta-information to guide the feature maps produced by the intermediate layers, giving them the properties needed for target tasks.  Examples include using segmentation loss or depth loss to improve the accuracy of object detectors.  Deep supervision is the most common type of auxiliary supervision.  It inserts additional prediction layers in the middle layers for training, such as the application of multi-layer decoders in transformer-based methods.|

---

## Streamlit Output🙌

![WhatsApp Image 2025-01-10 at 12 41 52_c9daaba1](https://github.com/user-attachments/assets/049addd1-cb29-4025-9666-fc2d09c1a44e)

![WhatsApp Image 2025-01-10 at 12 41 52_4ac690aa](https://github.com/user-attachments/assets/90b2d1bb-409b-4354-9fff-181312af22f9)


---

## Key Highlights 🌟

- **Robust RAG Workflow:** Combines retrieval-based search with generative answers.
- **Error Handling:** Alerts users if PDF processing or chat generation fails.
- **Data Security:** Uploaded PDFs are processed locally and deleted after use.

---

## Future Enhancements 🔮

- Add support for other file formats (e.g., Word, Excel).
- Enable deployment on cloud platforms for remote access.
- Implement multi-language support.

---

