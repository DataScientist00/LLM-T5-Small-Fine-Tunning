# 📄 Dialogue Summarizer using LLMs + Transformers

Everyday conversations, chats, and dialogues are full of information—but often unstructured and lengthy. This project brings the power of **large language models (LLMs)** to **automatically summarize dialogues**, using state-of-the-art **transformer models** fine-tuned on the **SAMSum dataset**.

Whether you're building a meeting summarizer, chatbot transcript shortener, or simply exploring conversational AI, this repo is a great starting point.

## 📺 Watch the Demo  
[![YouTube Video](https://img.shields.io/badge/YouTube-Watch%20Video-red?logo=youtube&logoColor=white&style=for-the-badge)](https://youtu.be/Dous6pBrYbc)

🧠 Dialogue Summarizer: Fine-tuned LLM on SAMSum Dataset using Hugging Face 🤗

![Image](https://github.com/user-attachments/assets/3cd4010a-65c6-493f-960a-9d5ea2bf0a36)

📚 What is This Project?

This project fine-tunes **Google's T5-small model** (and supports others like Flan-T5 and BART) on the **SAMSum dataset** to summarize informal conversations. It enables a model to take real-world dialogues and output clean, concise summaries.

💬 Sample Input:  
> Luffy: I want to be the Pirate King!  
> Naruto: That’s cool! I want to be Hokage someday.  
> Luffy: Let's train together!

✅ Output Summary:  
> Luffy and Naruto discussed their dreams and decided to train together.

---

🚀 Tech Stack

| Component              | Description                                                     |
|------------------------|-----------------------------------------------------------------|
| 🧠 LLM                 | T5-small (60M), Flan-T5, BART, Pegasus                          |
| 📚 Dataset             | SAMSum (from Hugging Face Datasets)                            |
| 🧩 Training            | Hugging Face Transformers + Trainer API                        |
| 📦 Tokenization        | AutoTokenizer + DataCollatorForSeq2Seq                         |
| 📊 Evaluation          | ROUGE (via built-in Trainer evaluation)                        |
| 🔍 Inference Pipeline  | Hugging Face `pipeline("summarization")` for easy prediction   |

🛠️ Features

✅ Fine-tunes LLMs for abstractive dialogue summarization  
✅ Supports different transformer models (T5, BART, Pegasus, etc.)  
✅ Inference-ready using Hugging Face `pipeline`  
✅ Lightweight and suitable for Kaggle notebooks (optimized for 1 GPU)  
✅ Clean modular code blocks for easy reuse

🧪 How It Works

1. Load the SAMSum dataset using `datasets.load_dataset`
2. Preprocess by prefixing input with `"summarize: "` (for T5-type models)
3. Tokenize dialogues and summaries with `AutoTokenizer`
4. Train using Hugging Face's `Trainer`
5. Generate summaries via `pipeline("summarization")`

🧑‍💻 Getting Started

1. Clone the repo  
   ```bash
   git clone https://github.com/DataScientist00/LLM-T5-Small-Fine-Tunning.git  
   cd LLM-T5-Small-Fine-Tunning
   ```

2. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) If running locally, ensure CUDA is set up for GPU training

4. Train the model  
   Run notebook or script to fine-tune the model on SAMSum

5. Inference  
   ```python
   from transformers import pipeline
   summarizer = pipeline("summarization", model="t5_samsum_finetuned_model")
   print(summarizer(your_dialogue_text))
   ```

💡 Example Usage

```python
sample = """
Naruto: I won the ramen contest again!
Luffy: No way! I ate 20 meat bones yesterday!
Naruto: We should team up!
"""

result = summarizer(sample, max_length=100, min_length=30)
print(result[0]['summary_text'])
```

🌍 Why This Project?

- Dialogue summarization is critical in meetings, chatbots, and virtual assistants.
- Finetuning on domain-specific datasets improves accuracy over generic models.
- SAMSum provides real-world style conversations with human-written summaries.
- T5 and BART are flexible, multilingual, and production-ready.

🧠 Future Enhancements

- Try larger models like `t5-base`, `bart-large`, `flan-t5-base`  
- Export to ONNX or TorchScript for deployment  
- Add Streamlit UI for live summarization  
- Integrate into WhatsApp/Telegram chatbot
- 

🙌 Acknowledgements

- Hugging Face Transformers  
- SAMSum dataset by K. Karthick on HF  
- Kaggle for free GPU compute  
- 🤗 Community for open-source LLMs

## 📞 Contact

- **Email**: nikzmishra@gmail.com  
- **Kaggle**: [@codingloading](https://www.kaggle.com/codingloading)  

---

⭐ If you like this project, don’t forget to **star** the repo and **share** it with your fellow ML friends!
