import nltk
from nltk.tokenize import TextTilingTokenizer
from textblob import TextBlob
from pdfminer.high_level import extract_text
from pdfminer.pdfpage import PDFPage
from tqdm import tqdm
from config import QA_GENERATOR_MODEL, SUMMARIZER_MODEL
import ollama
import torch
from transformers import pipeline
import json
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.high_level import extract_text

RAW_JSONL_OUTPUT = 'qa_results.jsonl'

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
   safe_append_jsonl(stream['message']['content'], RAW_JSONL_OUTPUT)

   
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

def safe_append_jsonl(raw_output, filename="qa_results.jsonl", failed_filename="qa_failed.jsonl"):
    """
    Append valid JSON objects (not arrays) to filename.
    If output is invalid JSON, dump raw content to failed_filename.
    """
    try:
        # Parse output (string → Python)
        if isinstance(raw_output, str):
            parsed = json.loads(raw_output)
        else:
            parsed = raw_output

        # If model wrapped output in a list, flatten it
        if isinstance(parsed, list):
            entries = parsed
        else:
            entries = [parsed]

    except json.JSONDecodeError:
        tqdm.write("⚠ JSON decode failed. Attempting cleanup...")
        cleaned = raw_output.strip()

        # Remove triple backtick code fences (``` / ```json etc.)
        lines = cleaned.splitlines()
        lines = [
            line for line in lines
            if not line.strip().lower().startswith("```")
        ]
        cleaned = "\n".join(lines).strip()

        try:
            parsed = json.loads(cleaned)
            entries = parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError as e:
            tqdm.write(f"❌ Still not valid JSON. Dumping to {failed_filename}. Error: {e}")
            with open(failed_filename, "a", encoding="utf-8") as f:
                json.dump({"raw_output": raw_output}, f, ensure_ascii=False)
                f.write("\n")
            return

    # ✅ Append each entry as its own JSONL line
    with open(filename, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")



def build_summary_tree(nodes, branch_factor = 2):
    while len(nodes) > 1:
        new_level = []
        #print(f"\n new level - total no of nodes = {len(nodes)}")
        for i in tqdm(range(0, len(nodes), branch_factor), desc="Processing text chunks"):
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

def process_pdf_into_root_chunks(ip_file, start_page=0, end_page=None, batch=1):
    if not ip_file.endswith(".pdf"):
        ip_file += ".pdf"

    tt = TextTilingTokenizer()
    root_segments = []

    # If end_page not given, process till last page
    if end_page is None:
        with open(ip_file, 'rb') as f:
            parser = PDFParser(f)
            doc = PDFDocument(parser)
            total_pages = sum(1 for _ in PDFPage.create_pages(doc))
            end_page = total_pages

    # Process in batches
    for batch_start in tqdm(range(start_page, end_page, batch), desc="Processing PDF pages in batches"):
        batch_end = min(batch_start + batch, end_page)
        page_range = list(range(batch_start, batch_end))

        # Extract text for the batch of pages
        cur_page_text = extract_text(ip_file, page_numbers=page_range)
        cur_page_text_striped = cur_page_text.strip()

        if not cur_page_text_striped:
            continue

        try:
            segments = [s.strip().replace('\n', ' ') for s in tt.tokenize(cur_page_text_striped)]
            root_segments.extend(segments)
            tqdm.write(f"[Pages {batch_start+1}-{batch_end}] Segments: {len(segments)}")
        except ValueError:
            tqdm.write(f"[Pages {batch_start+1}-{batch_end}] skipped.")

    root_objs = [TreeNode(text.strip()) for text in root_segments]
    return root_objs


def convert_to_alpaca(input_file="qa_results.jsonl", output_file="qa_results_alpaca.jsonl"):
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:
            try:
                qa = json.loads(line)
                alpaca_entry = {
                    "instruction": qa["question"],  # question directly as instruction
                    "input": "",                    # input left empty
                    "output": qa["answer"]          # answer stays as output
                }
                fout.write(json.dumps(alpaca_entry, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Skipping line due to error: {e}")

    print(f"✅ Converted file written to {output_file}")

def generate_qa_pairs(ip_doc_name, page_from, page_to, op_aplaca_doc_name):
    root_objects = process_pdf_into_root_chunks(ip_doc_name, page_from, page_to, 5)
    build_summary_tree(root_objects)
    convert_to_alpaca(RAW_JSONL_OUTPUT, op_aplaca_doc_name)


