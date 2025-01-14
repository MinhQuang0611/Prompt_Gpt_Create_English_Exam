from spellchecker import SpellChecker
from typing import Dict, List, Tuple

def initialize_spellchecker():
    """Initialize and return a SpellChecker instance"""
    return SpellChecker()

def check_text(text: str, spell: SpellChecker) -> List[Tuple[str, List[str], int]]:
    """
    Check text for spelling errors and return corrections
    
    Args:
        text (str): Text to check
        spell: SpellChecker instance
        
    Returns:
        List[Tuple[str, List[str], int]]: List of (wrong_word, suggestions, position)
    """
    words = text.split()
    corrections = []
    
    for i, word in enumerate(words):
        # Clean the word of punctuation for checking
        clean_word = ''.join(c for c in word.lower() if c.isalnum())
        
        if not spell.known([clean_word]):
            suggestions = list(spell.candidates(clean_word))
            corrections.append((word, suggestions, i))
    
    return corrections

def generate_correction_report(text: str, spell: SpellChecker) -> str:
    """
    Generate a detailed correction report
    
    Args:
        text (str): Text to analyze
        spell: SpellChecker instance
        
    Returns:
        str: Formatted correction report
    """
    corrections = check_text(text, spell)
    if not corrections:
        return "No spelling errors found! Great job!"
    
    report = "Spelling Correction Report:\n\n"
    original_words = text.split()
    
    for wrong_word, suggestions, position in corrections:
        context = " ".join(original_words[max(0, position-2):min(len(original_words), position+3)])
        report += f"Error found: '{wrong_word}'\n"
        report += f"Context: \"...{context}...\"\n"
        report += f"Suggested corrections: {', '.join(suggestions[:3])}\n"
        report += f"Position in text: Word #{position + 1}\n\n"
    
    return report

def add_custom_words(spell: SpellChecker, words: List[str]):
    """
    Add custom words to the dictionary (e.g., technical terms)
    
    Args:
        spell: SpellChecker instance
        words (List[str]): List of words to add
    """
    spell.word_frequency.load_words(words)

# Example usage
if __name__ == "__main__":
    # Initialize the spell checker
    spell = initialize_spellchecker()
    
    # Add some custom technical words
    add_custom_words(spell, ['pygame', 'numpy', 'pandas'])
    
    # Example text with errors
    sample_text = """The deadlist virus in modern history, perhaps of all time, was the 1918 Spanish Flu. It killed about 20 to 50 million people worldwide, perhaps more. The total death toll is unknown because medical records were not kept in many areas.
The pandemic hit during World War I and devastated military troops. In the United States, for instance, more servicemen were killed from the flu than from the war itself. The Spanish flu was fatal to a higher proportion of young adults than most flu viruses.
The pandemic started mildly, in the spring of 1918, but was followed by a much more severe wave in the fall of 1918. The war likely contributed to the devastating mortality numbers, as large outbreaks occurred in military forces living in close quarters. Poor nutrition and the unsanitary conditions of war camps had an effect.
A third wave occurred in the winter and spring of 1919, and a fourth, smaller wave occurred in a few areas in spring 1920. Initial symptoms of the flu were typical: sore throat, headache, and fever. The flu often progressed rapidly to cause severe pneumonia and sometimes hemorrhage in the lungs and mucus membranes. A characteristic feature of severe cases of the Spanish Flu was heliotrope cyanosis, where the patient’s face turned blue from lack of oxygen in the cells. Death usually followed within hours or days.
Modern medicine such as vaccines, antivirals, and antibiotics for secondary infections were not available at that time, so medical personnel couldn’t do much more than try to relieve symptoms.
The flu ended when it had infected enough people that those who were susceptible had either died or developed immunity."""
    
    # Get the correction report
    report = generate_correction_report(sample_text, spell)
    print(report)
    
    # If you want to get just the corrections and suggestions
    corrections = check_text(sample_text, spell)
    for wrong_word, suggestions, position in corrections:
        print(f"'{wrong_word}' might be spelled as: {suggestions}")