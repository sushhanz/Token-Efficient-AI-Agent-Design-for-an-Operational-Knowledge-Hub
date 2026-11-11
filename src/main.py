import warnings
import os
import time
import string
import re
from collections import Counter
from dotenv import load_dotenv
from datasets import load_dataset, Dataset, load_from_disk
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from sentence_transformers import CrossEncoder
from langchain_groq import ChatGroq

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv() # I have my (free) api key in .env

LOCAL_DATA_DIR = "./data_cache"


def normalize_answer(s):
    """Normalizes answer for use in F1 and Exact Match scoring."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, ground_truth):
    """Returns 1 if normalized prediction exactly matches normalized ground truth, else 0."""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction, ground_truth):
    """Computes token-level F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return float(pred_tokens == gt_tokens)
    
    if num_same == 0:
        return 0.0
    
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_prototyping_data(sample_size=100):
    """
    Downloads and caches a small subset locally if it a cached version is not available    
    """

    hotpot_path = os.path.join(LOCAL_DATA_DIR, "hotpot_qa")

    if os.path.exists(LOCAL_DATA_DIR):
        print("Loading cached datasets from local disk...")

        hotpot = load_from_disk(hotpot_path)
    else:
        print("First run: Streaming and saving subset locally...")
        os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

        hotpot_stream = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train", streaming=True)
        hotpot = Dataset.from_list(list(hotpot_stream.take(sample_size)))

        hotpot.save_to_disk(hotpot_path)
        print("Subsets saved to local disk!")

    return hotpot
def run_evaluation(name, pipeline_func, eval_set):
    """Evaluation loop"""
    total_em = 0
    total_f1 = 0.0
    total_latency = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    length = len(eval_set)
    
    for qa in eval_set:
        question = qa['question']
        ground_truth = qa['answer']

        start_time = time.perf_counter()
        
        ai_message = pipeline_func(question)
        
        latency = time.perf_counter() - start_time
        prediction = ai_message.content
        
        usage = getattr(ai_message, 'usage_metadata', {})
        input_tokens = usage.get('input_tokens', 0) if usage else 0
        output_tokens = usage.get('output_tokens', 0) if usage else 0
        # print(prediction+ "\n")
        # print(ground_truth+"\n")
        total_em += compute_exact_match(prediction, ground_truth)
        total_f1 += compute_f1(prediction, ground_truth)
        total_latency += latency
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

    print(f"\n================ FINAL BENCHMARK SUMMARY ({name}) ================")
    print(f"Exact Match Rate (EM) : {(total_em / length) * 100:.2f}%")
    print(f"Average F1 Score      : {total_f1 / length:.4f}")
    print(f"Average Latency       : {total_latency / length:.3f} seconds / query")
    print(f"Average Input Tokens  : {total_input_tokens / length:.1f} tokens / query")
    print(f"Average Output Tokens : {total_output_tokens / length:.1f} tokens / query")
    print(f"Total Tokens Processed: {total_input_tokens + total_output_tokens}")
    print("========================================================================\n")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    sample_items = get_prototyping_data(sample_size=51)

    documents = []
    questions_and_answers = []

    for item in sample_items:
        questions_and_answers.append({
            "question": item["question"],
            "answer": item["answer"]
        })
    
        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            documents.append(
                Document(
                    page_content=" ".join(sentences),
                    metadata={"source": "hotpot_qa", "title": title}
                )
            )

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0)
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    
    EVAL_LENGTH = 50
    budget_token = 84
    budget_word = 15
    eval_set = questions_and_answers[1:EVAL_LENGTH + 1] # I used the first question as an example in the prompt

    vanilla_prompt = ChatPromptTemplate.from_template(
        "You are an expert AI assistant answering questions based on the retrieved context.\n"
        "Answer the question with a short answer.\n"
        "For example:\n"
        "Example question: Which magazine was started first Arthur's Magazine or First for Women?\n"
        "Example Answer: Arthur's Magazine\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    )   
    budget_prompt_token = ChatPromptTemplate.from_template(
        "You are an expert AI assistant answering questions based on the retrieved context.\n"
        f"Answer the question with a short answer and use less than {budget_token} tokens.\n"
        "For example:\n"
        "Example question: Which magazine was started first Arthur's Magazine or First for Women?\n"
        "Example Answer: Arthur's Magazine\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    )
    budget_prompt_word = ChatPromptTemplate.from_template(
        "You are an expert AI assistant answering questions based on the retrieved context.\n"
        f"Answer the question with a short answer and use less than {budget_word} words.\n"
        "For example:\n"
        "Example question: Which magazine was started first Arthur's Magazine or First for Women?\n"
        "Example Answer: Arthur's Magazine\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    )

    
    vanilla_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | vanilla_prompt | llm)
    budget_chain_token = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | budget_prompt_token | llm)
    budget_chain_word = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | budget_prompt_word | llm)



    def get_compressed_context(question, top_n=4):
        retrieved_docs = retriever.invoke(question)
        raw_sentences = []
        for doc in retrieved_docs:
            sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', doc.page_content) if len(s.strip()) > 10]
            raw_sentences.extend(sents)
        
        unique_sentences = list(dict.fromkeys(raw_sentences))
        pairs = [[question, sentence] for sentence in unique_sentences]
        scores = cross_encoder.predict(pairs)
        scored_sentences = sorted(zip(scores, unique_sentences), key=lambda x: x[0], reverse=True)
        top_sentences = [s for score, s in scored_sentences[:top_n]]
        return "\n".join(top_sentences)


    def invoke_context_compressed(q):
        context_str = get_compressed_context(q)
        prompt = vanilla_prompt.format_messages(context=context_str, question=q)
        return llm.invoke(prompt)


    def invoke_fully_compressed_token(q):
        context_str = get_compressed_context(q)
        prompt = budget_prompt_token.format_messages(context=context_str, question=q)
        return llm.invoke(prompt)
    
    def invoke_fully_compressed_word(q):
            context_str = get_compressed_context(q)
            prompt = budget_prompt_word.format_messages(context=context_str, question=q)
            return llm.invoke(prompt)


    
    run_evaluation("Vanilla", lambda q: vanilla_chain.invoke(q), eval_set)
    run_evaluation("Reasoning Budget (token)", lambda q: budget_chain_token.invoke(q), eval_set)
    # run_evaluation("Reasoning Budget (words)", lambda q: budget_chain_word.invoke(q), eval_set)
    run_evaluation("Context Compressed", invoke_context_compressed, eval_set)
    run_evaluation("Fully Compressed (token)", invoke_fully_compressed_token, eval_set)
    # run_evaluation("Fully Compressed (word)", invoke_fully_compressed_word, eval_set)

    """
    I attempted to use words instead of tokens to reduce the output tokens. However I got rate limited before I could fully test this.
    Due to time constraints and the scope of this exercise I left this part out of the report.
    
    """


    


if __name__ == "__main__":
    main()