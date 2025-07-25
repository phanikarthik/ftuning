import nltk
from nltk.tokenize import TextTilingTokenizer

sample_text = """
The sun is a star located at the center of the solar system. It provides light and heat to Earth and other planets. Solar flares can affect satellites and communication systems on Earth.

Planets revolve around the sun in elliptical orbits. Mercury is the closest planet to the sun, while Neptune is the farthest. Some planets have many moons.

Elephants are the largest land mammals. They are known for their intelligence and social behavior. A herd of elephants is led by a matriarch, typically the oldest female.

Lions are carnivorous animals found in Africa and parts of India. They live in groups called prides. Male lions have a distinctive mane and often defend the pride from intruders.
"""

def main():
   #nltk.download('all')
   print("NLTK is working!")
   # Initialize the tokenizer
   tt = TextTilingTokenizer()

   # Apply the tokenizer
   segments = tt.tokenize(sample_text)

   # Print results
   for i, segment in enumerate(segments):
     print(f"\n--- Segment {i+1} ---\n{segment.strip()}")

if __name__ == "__main__":
    main()