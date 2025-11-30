from qa_generator import generate_qa_pairs
from config import INPUT_FILE_NAME, STAGE1_OUTPUT
import sys
import winsound

def sanitize_arg(arg, default=0):
    """Convert arg to int safely, else return default."""
    try:
        return int(arg)
    except (ValueError, TypeError):
        return default

def main():
   #nltk.download('all')
   #generate_qa_pairs(INPUT_FILE_NAME, 3, 15, STAGE1_OUTPUT)

   #if len(sys.argv) < 3:
   #   print("Usage: python script.py <start_page> <end_page>")
   #   sys.exit(1)

   start_page = 417 #sanitize_arg(sys.argv[1], default=0)
   end_page = 424 #sanitize_arg(sys.argv[2], default=None)
   
   end_page = end_page - 1

   print(f"\n Processing from page numbers {start_page} to {end_page}")
   generate_qa_pairs(INPUT_FILE_NAME, start_page, end_page, STAGE1_OUTPUT)

   # Play a beep at 500 Hz for 4000 milliseconds
   winsound.Beep(500, 6000)



if __name__ == "__main__":
    main()