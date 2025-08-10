import nltk
from nltk.tokenize import TextTilingTokenizer
from textblob import TextBlob
from pdfminer.high_level import extract_text
from pdfminer.pdfpage import PDFPage
from tqdm import tqdm
from config import INPUT_FILE_NAME, QA_GENERATOR_MODEL, SUMMARIZER_MODEL
import ollama
import torch
from transformers import pipeline
import json

#pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-alpha", torch_dtype=torch.bfloat16, device_map="auto")

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
headers = {"Authorization": f"Bearer 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"}

class TreeNode:
    def __init__(self, content):
        self.content = content
        self.children = []  # holds multiple children
        #self.qa_dict = {}  # dict with question as key and answer as value

def chunk_text_with_overlap(text_with_newline, doOverlapping = True, chunk_size=1000, overlap_size=200):
    """
    Chunks text into overlapping blocks using sentence boundaries from TextBlob.
    
    Args:
        text (str): The input text.
        chunk_size (int): Max characters in a chunk.
        overlap_size (int): Number of characters to overlap.

    Returns:
        List[str]: A list of text chunks.
    """
    text = text_with_newline.replace('\n', ' ')
    blob = TextBlob(text)
    sentences = [str(s) for s in blob.sentences]

    chunks = []
    current_chunk = ""

    if(doOverlapping):
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 > chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = current_chunk[-overlap_size:] + " " + sentence
            else:
                current_chunk += " " + sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        return chunks
    else:
        return sentences

def generate_qa_for_node(node):
   instruction_prompt = f"""
   Context: {node.content}"""

   stream = ollama.chat(
   model=QA_GENERATOR_MODEL,
   messages=[
     {'role': 'user', 'content': instruction_prompt},
   ],
   stream=False,
   )

   #print(stream['message']['content'])
   safe_append_jsonl(stream['message']['content'])

   
def summarize(text):
   instruction_prompt = f"""
   Context: {text}"""

   stream = ollama.chat(
   model=SUMMARIZER_MODEL,
   messages=[
     {'role': 'user', 'content': instruction_prompt},
   ],
   stream=False,
   )

   return stream['message']['content']
   #print(stream['message']['content'])

import json

def safe_append_jsonl(raw_output, filename="qa_results.jsonl", failed_filename="qa_failed.jsonl"):
    """
    Append valid JSON to filename, dump invalid entries to failed_filename.
    """
    try:
        # Direct parse attempt
        if isinstance(raw_output, str):
            parsed = json.loads(raw_output)
        else:
            parsed = raw_output

    except json.JSONDecodeError:
        print("⚠ JSON decode failed. Attempting cleanup...")
        cleaned = raw_output.strip()

        # Remove triple backtick code fences with or without json keyword
        lines = cleaned.splitlines()
        lines = [
            line for line in lines
            if not line.strip().lower().startswith("```")  # removes ``` and ```json etc.
        ]
        cleaned = "\n".join(lines).strip()

        # Retry parsing after cleaning
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"❌ Still not valid JSON. Dumping to {failed_filename}. Error: {e}")

            # Save raw output to failed file for debugging
            with open(failed_filename, "a", encoding="utf-8") as f:
                # Store as plain JSON string entry for consistency
                json.dump({"raw_output": raw_output}, f, ensure_ascii=False)
                f.write("\n")
            return  # Skip saving to main file

    # If parsing worked, append to main file
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(parsed, ensure_ascii=False) + "\n")


def build_summary_tree(nodes, branch_factor = 2):
    while len(nodes) > 1:
        new_level = []
        #print(f"\n new level - total no of nodes = {len(nodes)}")
        for i in range(0, len(nodes), branch_factor):
            children = nodes[i:i + branch_factor]
            if len(children) == 1:
                new_level.append(children[0])
                continue

            combined_text = " ".join(child.content for child in children)
            summary = summarize(combined_text)
            parent = TreeNode(summary)
            generate_qa_for_node(parent)
            parent.children = children
            #print(f"\n--- Segment {i}, {i + 1} ---\n{parent.content}")
            new_level.append(parent)

        nodes = new_level
    return nodes[0]

root_segments = []
def process_pdf_into_root_chunks(ip_file, start_page = 0, end_page = 1):

  if not ip_file.endswith(".pdf"):
    ip_file += ".pdf"
  
  tt = TextTilingTokenizer()

  with open(ip_file, 'rb') as f:

    page_range = list(range(start_page, end_page))
    cur_page_text = extract_text(ip_file, page_numbers=page_range)  # 0-based index
    cur_page_text_striped = cur_page_text.strip()
    
    try:
        segments = [s.strip().replace('\n', ' ') for s in tt.tokenize(cur_page_text_striped)]
        #segments = tt.tokenize(cur_page_text_striped)
        root_segments.extend(segments)  # Correct way to accumulate
           
        # Print results
        for i, segment in enumerate(segments):
            print(f"\n--- Segment {i+1} ---\n{segment.strip()}")
    
    except ValueError:
        pass

    root_objs = [TreeNode(text.strip()) for text in root_segments]
    return root_objs


   
def main():
   #nltk.download('all')
   root_objects = process_pdf_into_root_chunks(INPUT_FILE_NAME, 3, 5)
   
   #for node in root_objects:
   #    generate_qa_for_node(node)

   build_summary_tree(root_objects)
   #leaf_nodes = [TreeNode(content=seg.strip()) for seg in root_segments]
   # Initialize the tokenizer
   #tt = TextTilingTokenizer()

   # Apply the tokenizer
   #segments = tt.tokenize(sample_text)

   # Print results
   #for i, segment in enumerate(segments):
   #  print(f"\n--- Segment {i+1} ---\n{segment.strip()}")

if __name__ == "__main__":
    main()