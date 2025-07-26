import nltk
from nltk.tokenize import TextTilingTokenizer
from textblob import TextBlob
from pdfminer.high_level import extract_text
from pdfminer.pdfpage import PDFPage
from tqdm import tqdm
from config import INPUT_FILE_NAME

sample_text = """
The sun is a star located at the center of the solar system. It provides light and heat to Earth and other planets. Solar flares can affect satellites and communication systems on Earth.

Planets revolve around the sun in elliptical orbits. Mercury is the closest planet to the sun, while Neptune is the farthest. Some planets have many moons.

Elephants are the largest land mammals. They are known for their intelligence and social behavior. A herd of elephants is led by a matriarch, typically the oldest female.

Lions are carnivorous animals found in Africa and parts of India. They live in groups called prides. Male lions have a distinctive mane and often defend the pride from intruders.
"""

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

root_segments = []
def process_pdf_in_page_range(ip_file, start_page = 0, end_page = 1):

  if not ip_file.endswith(".pdf"):
    ip_file += ".pdf"
  
  tt = TextTilingTokenizer()

  #page_texts = []
  all_pages_sentences = []
  cur_page_sentences = []
  with open(ip_file, 'rb') as f:

    page_range = list(range(start_page, end_page))
    cur_page_text = extract_text(ip_file, page_numbers=page_range)  # 0-based index
    cur_page_text_striped = cur_page_text.strip()
    
    try:
        segments = tt.tokenize(cur_page_text_striped)
        root_segments.extend(segments)  # Correct way to accumulate
           
        # Print results
        for i, segment in enumerate(segments):
            print(f"\n--- Segment {i+1} ---\n{segment.strip()}")
    
    except ValueError:
        pass

   
def main():
   #nltk.download('all')
   print("NLTK is working!")
   process_pdf_in_page_range(INPUT_FILE_NAME, 3, 5)
   # Initialize the tokenizer
   #tt = TextTilingTokenizer()

   # Apply the tokenizer
   #segments = tt.tokenize(sample_text)

   # Print results
   #for i, segment in enumerate(segments):
   #  print(f"\n--- Segment {i+1} ---\n{segment.strip()}")

if __name__ == "__main__":
    main()