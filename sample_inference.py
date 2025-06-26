from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Load the fine-tuned model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("t5_samsum_finetuned_model")
tokenizer = AutoTokenizer.from_pretrained("t5_samsum_tokenizer")

# Create a pipeline
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

# Sample input
sample_text = '''Luffy: Naruto! You won the ramen eating contest again?! That’s your fifth win this month!...'''

# Run inference
summary = summarizer(sample_text, max_length=100, min_length=30, do_sample=False)
print("Summary:", summary[0]['summary_text'])
