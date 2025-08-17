from qa_generator import generate_qa_pairs
from config import INPUT_FILE_NAME, STAGE1_OUTPUT
import sys

def sanitize_arg(arg, default=0):
    """Convert arg to int safely, else return default."""
    try:
        return int(arg)
    except (ValueError, TypeError):
        return default

def main():
   #nltk.download('all')
   if len(sys.argv) < 3:
      print("Usage: python script.py <start_page> <end_page>")
      sys.exit(1)

   start_page = sanitize_arg(sys.argv[1], default=0)
   end_page = sanitize_arg(sys.argv[2], default=None)
   generate_qa_pairs(INPUT_FILE_NAME, start_page, end_page, STAGE1_OUTPUT)


if __name__ == "__main__":
    main()