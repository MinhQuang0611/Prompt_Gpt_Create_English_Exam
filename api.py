from flask import Flask, request, jsonify
from textblob import TextBlob
from spellchecker import SpellChecker
import re
import requests
import time
from gramformer import Gramformer
import torch

#  TEXTBLOB API
class TextBlobChecker:
    def __init__(self):
        pass
    def check_text(self, text):
        blob = TextBlob(text)
        sentences = text.split()
        corrections =[]
        word_position = 1

        for sentence in sentences:
            if not sentence.strip():
                continue
            sentence_blob = TextBlob(sentence)

            original_words = re.findall(r'\b\w+\b', sentence)
            corrections_words = re.findall(r'\b\w+\b', str(sentence_blob.correct()))

            for i, (orig, corr) in enumerate(zip(original_words, corrections_words)):
                if orig.lower() != corr.lower():
                    tags = dict(sentence_blob.tags)
                    word_type = tags.get(orig, 'unknown')

                    correction = {
                        'word': orig,
                        "suggestions" : corr,
                        "position" : word_position + i,
                        "word_type" : word_type,
                        'confidence' : self._caculate_confidence(orig, corr)
                    }
                    corrections.append(correction)

            word_position += len(original_words)
        return corrections
    def _caculate_confidence(self, original, correction):

        distance = self._levenshtien_distance(original.lower(), correction.lower())
        max_len = max(len(original), len(correction))

        confidence = (1 - (distance / max_len)) * 100

        return round(confidence, 2)


    def _levenshtien_distance (self, s1, s2):

        if len(s1) < len(s2):
            return self._levenshtien_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i+1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j+1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)


                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]
#  LANGUAGETOOL API
class LanguageTool:
    def __init__(self, language="en-US" ):
        self.language = language
        self.api_urls = "https://api.languagetool.org/v2/check"
    def check_grammar(self, text):
        "Check loi ngu phap va  chinh ta"
        params = {
            'text' : text,
            'language' : self.language,
            'enableOnly' : False
        }

        try:
            print("connnet to language toool")
            response = requests.post(url = self.api_urls, data = params, timeout = 10)
            response.raise_for_status()
            result = response.json()
            errors = []
            for match in result.get('matches', []):
                error = {
                    'message' : match['message'],
                    'context' : match['context']['text'],
                    'suggestions' : [fix['value'] for fix in match.get('replacements', [])[:3]],
                    'rule' : match['rule']["description"]
                
                }
                errors.append(error)
            return errors
        except requests.exceptions.RequestException as e:
            print(f"co loi xay ra : {str(e)}")
            return
    
    def print_errors(self, errors):
        if errors is None:
            print("khong ket noi duoc voi api")
            return
        if not errors:
            print("khong co loi duoc phat hien")
            return
        print("tim duoc loi trong van ban")
        for i, error in enumerate(errors, 1):
            print(f"Loi {i} :")
            print(F"issue : {error['message']}")
            print(f"content : {error['content']}")
            if error['suggestions']:
                print(f"suggestion : {', '.join(error['suggestions'])}")
            print(f"rule: {error['rule']}")
#  SPELLCHECKER API
class SpellingChecker:
    def __init__(self):
        self.spell = SpellChecker()
    
    def check_text(self, text):
        """
        Check spelling in a text and return corrections
        """
        words = text.split()
        word_positions = {}
        
        for index, word in enumerate(words):
            clean_word = word.strip('.,!?:;()[]{}""''')
            if clean_word not in word_positions:
                word_positions[clean_word] = []
            word_positions[clean_word].append(index)
        
        misspelled = self.spell.unknown([word.strip('.,!?:;()[]{}""''') for word in words])
        
        corrections = []
        for word in misspelled:
            positions = word_positions.get(word, [])
            
            correction = {
                'word': word,
                'suggestions': list(self.spell.candidates(word)),
                'positions': positions  
            }
            corrections.append(correction)
            
        return corrections

    def print_corrections(self, text):
        """
        Print spelling corrections in a formatted way
        """
        try:
            corrections = self.check_text(text)
            
            if not corrections:
                print("No spelling mistakes found!")
                return
            
            print(f"Found {len(corrections)} spelling mistakes:\n")
            
            for i, correction in enumerate(corrections, 1):
                print(f"Mistake #{i}:")
                print(f"- Word: {correction['word']}")
                print(f"- Suggestions: {', '.join(correction['suggestions'][:5])}")
                positions_str = ', '.join(str(pos + 1) for pos in correction['positions'])
                print(f"- Position(s): word #{positions_str}")
                print()
                
        except Exception as e:
            print(f"An error occurred while checking the text: {str(e)}")
# GRAMFOMER API
class GrammarChecker:
    def __init__(self, use_gpu=False):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.gf = self.setup_gramformer()

    def setup_gramformer(self):
        try:
            gf = Gramformer(models=1, use_gpu=True)
            return gf
        except Exception as e:
            print(f"Lỗi khi khởi tạo Gramformer: {str(e)}")
            return None

    def check_grammar(self, text):
        if self.gf is None:
            return {
                'original': text,
                'corrections': [],
                'suggestions': [],
                'error': 'Gramformer chưa được khởi tạo đúng cách'
            }

        try:
            results = {
                'original': text,
                'corrections': [],
                'suggestions': []
            }

            corrections = self.gf.correct(text, max_candidates=2)

            for correction in corrections:
                results['corrections'].append(correction)

                if correction != text:
                    differences = self.get_differences(text, correction)
                    results['suggestions'].extend(differences)

            return results
        except Exception as e:
            return {
                'original': text,
                'corrections': [],
                'suggestions': [],
                'error': f'Lỗi khi kiểm tra ngữ pháp: {str(e)}'
            }

    def get_differences(self, original, corrected):
        suggestions = []
        orig_words = original.split()
        corr_words = corrected.split()

        for i, (orig, corr) in enumerate(zip(orig_words, corr_words)):
            if orig != corr:
                suggestions.append(f"Nên thay '{orig}' bằng '{corr}'")

        return suggestions
    
app = Flask(__name__)
textBlob = TextBlobChecker()
languageTool = LanguageTool()
spell_checker = SpellingChecker()
gramformer = GrammarChecker()
@app.route('/check/textblob', methods =['POST'])
def textblob_check():
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({
                'error' : "thieu truong du lieu"
            }), 400
        text = data['text']
        corrections = textBlob.check_text(text)
        return jsonify ( {
            'success' : True,
            'corrections' : corrections,
            'total_issues' : len(corrections)
        }), 200
    except Exception as e:
        return jsonify({
            'error' : str(e),

        }), 500
@app.route('/check/languagetool', methods = ['POST'])
def language_check():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({
            'error' : "thieu truong du lieu"
        }), 400
    text = data['text']
    errors = languageTool.check_grammar(text)
    if errors is None:
        return jsonify({
            'error' : "khong the ket noi voi language tool api"
        }), 500
    return jsonify({"errors": errors}), 200   
@app.route('/check/spellchecker', methods = ["POST"])
def spell_check():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({
            'error' : "thieu truong du lieu"
        }), 400
    text = data['text']
    errors = spell_checker.check_text(text)
    return jsonify({
        "errors" : errors
    }), 200
@app.route('/check/gramformer', methods = ['POST'])
def gramformer_check():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({
            "error" : "thieu truong du lieu"
        }), 400
    text = data['text']
    errors = gramformer.check_grammar(text)
    return jsonify ({
        "errors" : errors
    }), 200
if __name__ == "__main__" :
    app.run(debug=True)