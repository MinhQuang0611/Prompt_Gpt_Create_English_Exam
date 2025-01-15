import spacy
from nltk.corpus import wordnet
from nltk.metrics import edit_distance
import nltk
from typing import List, Dict

# Download required NLTK data
nltk.download('wordnet')
nltk.download('words')

# Load spaCy model and NLTK words once
nlp = spacy.load('en_core_web_sm')
word_list = set(nltk.corpus.words.words())

def get_similar_words(word: str, max_distance: int = 2) -> List[str]:
    """Get words with similar spelling using edit distance."""
    return [w for w in word_list 
            if edit_distance(word, w) <= max_distance 
            and w != word][:5]

def get_synonyms(word: str) -> List[str]:
    """Get synonyms using WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name() != word:
                synonyms.add(lemma.name())
    return list(synonyms)[:5]

def get_context_based_suggestions(text: str, word: str) -> List[str]:
    """Get suggestions based on context using spaCy."""
    doc = nlp(text)
    suggestions = []
    
    # Find the target word's position and properties
    target_token = None
    for token in doc:
        if token.text.lower() == word.lower():
            target_token = token
            break
    
    if target_token:
        # Get words with similar POS tag and dependencies
        for token in doc:
            if (token.pos_ == target_token.pos_ and 
                token.dep_ == target_token.dep_ and 
                token.text.lower() != word.lower()):
                suggestions.append(token.text)
    
    return suggestions[:5]

def get_suggestions(text: str, word: str) -> Dict[str, List[str]]:
    """Get comprehensive word suggestions."""
    similar_words = get_similar_words(word)
    synonyms = get_synonyms(word)
    context_suggestions = get_context_based_suggestions(text, word)
    
    return {
        'similar_spelling': similar_words,
        'synonyms': synonyms,
        'context_based': context_suggestions
    }

# Example usage
def main():
    text = "The cat quickly jumped over the fence."
    word = "quickly"
    
    suggestions = get_suggestions(text, word)
    
    print(f"Suggestions for '{word}' in context: '{text}'\n")
    for category, words in suggestions.items():
        print(f"{category.replace('_', ' ').title()}:")
        print(", ".join(words))
        print()

if __name__ == "__main__":
    main()