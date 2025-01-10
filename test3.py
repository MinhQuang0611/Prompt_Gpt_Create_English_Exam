# First part with Flair (your existing code)
from flair.data import Sentence
from flair.models import SequenceTagger

# load tagger
tagger = SequenceTagger.load("flair/upos-english")

# make example sentence
sentence = Sentence("I love Berlin.")

# predict NER tags
tagger.predict(sentence)

# print sentence
print(sentence)

# print predicted NER tags
print('The following NER tags are found:')
# iterate over entities and print
for entity in sentence.get_spans('pos'):
    print(entity)

# Now let's add spaCy analysis
import spacy
from spacy import displacy

# Load English language model
nlp = spacy.load("en_core_web_sm")

# Process the same text with spaCy
doc = nlp("I love Berlin.")

# Print dependency information
print("\nDependency Parse:")
for token in doc:
    print(f"{token.text:10} {token.dep_:10} {token.head.text}")

# Generate and display syntax tree visualization
# Note: This will create an HTML visualization
options = {"compact": True, "bg": "#ffffff", "color": "#000000", "font": "Arial"}
html = displacy.render(doc, style="dep", options=options)

# You can save the visualization to a file
with open("syntax_tree.html", "w", encoding="utf-8") as f:
    f.write(html)

# To print detailed syntactic information
print("\nDetailed Syntactic Analysis:")
for token in doc:
    print(f"""
Token: {token.text}
    Dependency: {token.dep_}
    Head word: {token.head.text}
    Part of speech: {token.pos_}
    Syntactic tag: {token.tag_}
    """)