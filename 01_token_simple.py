wikipedia = '''Natural language processing (NLP) is a subfield of linguistics,
computer science, and artificial intelligence concerned with the interactions between
computers and human language, in particular how to program computers to process and
analyze large amounts of natural language data. The result is a computer capable of
"understanding" the contents of documents, including the contextual nuances of the
language within them. The technology can then accurately extract information and
insights contained in the documents as well as categorize and organize the documents themselves.

Challenges in natural language processing frequently involve speech recognition,
natural language understanding, and natural-language generation.'''

tokens = [word.strip(' .,?!:;()"') for line in wikipedia.splitlines() for word in line.split(' ')]
print(tokens) # ['Natural', 'language', 'processing', 'NLP', 'is', ... ]