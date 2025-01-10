from textblob import TextBlob
import re

class TextBlobChecker:
    def __init__(self):
        pass
    
    def check_text(self, text):
        """
        Check text using TextBlob with enhanced error detection
        """
        blob = TextBlob(text)
        
        # Split into sentences for better analysis
        sentences = text.split('.')
        corrections = []
        
        # Process each sentence
        word_position = 1  # Keep track of global word position
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Create TextBlob for this sentence
            sentence_blob = TextBlob(sentence)
            
            # Get original and corrected words
            original_words = re.findall(r'\b\w+\b', sentence)
            corrected_words = re.findall(r'\b\w+\b', str(sentence_blob.correct()))
            
            # Compare words and find differences
            for i, (orig, corr) in enumerate(zip(original_words, corrected_words)):
                if orig.lower() != corr.lower():
                    # Get word tags for context
                    tags = dict(sentence_blob.tags)
                    word_type = tags.get(orig, 'unknown')
                    
                    correction = {
                        'word': orig,
                        'suggestion': corr,
                        'position': word_position + i,
                        'word_type': word_type,
                        'confidence': self._calculate_confidence(orig, corr)
                    }
                    corrections.append(correction)
            
            word_position += len(original_words)
        
        return corrections
    
    def _calculate_confidence(self, original, correction):
        """
        Calculate a simple confidence score for the correction
        """
        # Calculate Levenshtein distance
        distance = self._levenshtein_distance(original.lower(), correction.lower())
        max_len = max(len(original), len(correction))
        
        # Convert to a confidence score (0-100)
        confidence = (1 - (distance / max_len)) * 100
        return round(confidence, 2)
    
    def _levenshtein_distance(self, s1, s2):
        """
        Calculate the Levenshtein distance between two strings
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]
    
    def print_corrections(self, text):
        """
        Print corrections in a formatted way
        """
        try:
            corrections = self.check_text(text)
            
            if not corrections:
                print("No spelling or grammar issues found!")
                return
            
            print(f"Found {len(corrections)} potential issues:\n")
            
            for i, correction in enumerate(corrections, 1):
                print(f"Issue #{i}:")
                print(f"- Word: {correction['word']}")
                print(f"- Suggestion: {correction['suggestion']}")
                print(f"- Position: word #{correction['position']}")
                print(f"- Word Type: {correction['word_type']}")
                print(f"- Confidence: {correction['confidence']}%")
                print()
                
        except Exception as e:
            print(f"An error occurred while checking the text: {str(e)}")

# Example usage
if __name__ == "__main__":
    checker = TextBlobChecker()
    
    test_text = """The deadlist virus in modern history, perhaps of all time, was the 1918 Spanish Flu. 
    It killed about 20 to 50 million people worldwide, perhaps more. 
    The total death toll is unknown because medical records were not kept in many areas."""
    
    print("Checking text:")
    print(test_text)
    print("\nResults:")
    checker.print_corrections(test_text)