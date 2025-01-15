from flask import Flask, request, jsonify
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from spellchecker import SpellChecker
import re
import requests
from underthesea import ner
import spacy
from typing import Dict, List
from gramformer import Gramformer
import torch
from functools import wraps
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom Exceptions
class GrammarCheckerException(Exception):
    """Base exception for grammar checker errors"""
    pass

class ValidationError(GrammarCheckerException):
    """Raised when input validation fails"""
    pass

# Request Validation Decorator
def validate_request(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        data = request.get_json()
        if not data or 'text' not in data:
            raise ValidationError("Missing required field: 'text'")
        if not isinstance(data['text'], str):
            raise ValidationError("Field 'text' must be a string")
        return f(*args, **kwargs)
    return decorated_function

# Response Models
@dataclass
class CorrectionResponse:
    success: bool
    data: Any
    total_issues: int = 0
    error: Optional[str] = None

# Base Grammar Checker Interface
class BaseGrammarChecker(ABC):
    @abstractmethod
    def check_text(self, text: str) -> Dict:
        pass

# TextBlob Implementation
# class TextBlobChecker(BaseGrammarChecker):
#     def check_text(self, text: str) -> List[Dict]:
#         blob = TextBlob(text)
#         sentences = text.split()
#         corrections = []
#         word_position = 1

#         for sentence in sentences:
#             if not sentence.strip():
#                 continue
            
#             sentence_blob = TextBlob(sentence)
#             original_words = re.findall(r'\b\w+\b', sentence)
#             corrected_words = re.findall(r'\b\w+\b', str(sentence_blob.correct()))

#             for i, (orig, corr) in enumerate(zip(original_words, corrected_words)):
#                 if orig.lower() != corr.lower():
#                     tags = dict(sentence_blob.tags)
#                     corrections.append({
#                         'word': orig,
#                         'suggestions': corr,
#                         'position': word_position + i,
#                         'word_type': tags.get(orig, 'unknown'),
#                         'confidence': self._calculate_confidence(orig, corr)
#                     })

#             word_position += len(original_words)
#         return corrections

#     def _calculate_confidence(self, original: str, correction: str) -> float:
#         distance = self._levenshtein_distance(original.lower(), correction.lower())
#         max_len = max(len(original), len(correction))
#         return round((1 - (distance / max_len)) * 100, 2)

#     def _levenshtein_distance(self, s1: str, s2: str) -> int:
#         if len(s1) < len(s2):
#             return self._levenshtein_distance(s2, s1)
#         if len(s2) == 0:
#             return len(s1)

#         previous_row = range(len(s2) + 1)
#         for i, c1 in enumerate(s1):
#             current_row = [i + 1]
#             for j, c2 in enumerate(s2):
#                 current_row.append(min(
#                     previous_row[j + 1] + 1,  # insertion
#                     current_row[j] + 1,       # deletion
#                     previous_row[j] + (c1 != c2)  # substitution
#                 ))
#             previous_row = current_row

#         return previous_row[-1]

# LanguageTool Implementation
class LanguageToolChecker(BaseGrammarChecker):
    def __init__(self, language: str = "en-US"):
        self.language = language
        self.api_url = "https://api.languagetool.org/v2/check"
        # Load NLP models for both English and Vietnamese
        try:
            self.nlp_vi = spacy.load("vi_core_news_lg")
        except:
            logger.warning("Could not load Vietnamese NLP model")
            self.nlp_vi = None

    def is_potential_vietnamese_name(self, text: str) -> bool:
        """
        Kiểm tra xem một từ có khả năng là tên người Việt Nam không dựa trên các quy tắc:
        - Viết hoa chữ cái đầu
        - Độ dài phù hợp với tên Việt Nam
        - Không chứa ký tự đặc biệt
        - Có dấu tiếng Việt hợp lệ
        """
        if not text or not text[0].isupper():
            return False

        # Kiểm tra độ dài hợp lý cho tên Việt Nam (thường từ 2-5 ký tự)
        if len(text.strip()) < 2 or len(text.strip()) > 5:
            return False

        # Kiểm tra các ký tự hợp lệ trong tên tiếng Việt
        vietnamese_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ áàảãạâấầẩẫậăắằẳẵặéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđĐ')
        return all(c in vietnamese_chars for c in text)

    def extract_names_underthesea(self, text: str) -> List[str]:
        """
        Sử dụng Underthesea để nhận diện tên người từ văn bản tiếng Việt.
        Được cải thiện để xử lý nhiều trường hợp hơn.
        """
        try:
            entities = ner(text)
            names = []
            current_name = []
            
            for token, tag in zip(entities, [entity[3] for entity in entities]):
                if tag == 'PER':
                    current_name.append(token[0])
                elif current_name:
                    names.append(' '.join(current_name))
                    current_name = []
            
            if current_name:  # Add the last name if exists
                names.append(' '.join(current_name))
                
            return names
        except Exception as e:
            logger.error(f"Error in Underthesea name extraction: {str(e)}")
            return []

    def extract_potential_names(self, text: str) -> List[str]:
        """
        Kết hợp nhiều phương pháp để nhận diện tên người:
        1. Sử dụng Underthesea NER
        2. Phân tích cấu trúc câu để tìm proper nouns
        3. Áp dụng quy tắc nhận diện tên tiếng Việt
        """
        names = set()
        
        # 1. Sử dụng Underthesea
        names.update(self.extract_names_underthesea(text))
        
        # 2. Phân tích từ viết hoa trong câu
        words = text.split()
        for i, word in enumerate(words):
            word = word.strip('.,!?():;"\'')
            
            # Kiểm tra từ hiện tại
            if self.is_potential_vietnamese_name(word):
                # Kiểm tra context xung quanh để xác định có phải tên người không
                if i > 0:
                    prev_word = words[i-1].strip('.,!?():;"\'').lower()
                    # Các từ chỉ danh xưng hoặc họ thường đứng trước tên
                    name_indicators = {'anh', 'chị', 'cô', 'chú', 'bác', 'ông', 'bà', 'em', 'mr', 'ms', 'mrs'}
                    if prev_word in name_indicators:
                        names.add(word)
                        continue
                
                # Nếu là một phần của chuỗi tên người
                if i > 0 and self.is_potential_vietnamese_name(words[i-1]):
                    names.add(f"{words[i-1]} {word}")
                
                # Thêm vào danh sách tên tiềm năng nếu thỏa mãn điều kiện
                if self.is_potential_vietnamese_name(word):
                    names.add(word)

        # 3. Sử dụng SpaCy nếu có
        if self.nlp_vi and self.language == "vi":
            doc = self.nlp_vi(text)
            for ent in doc.ents:
                if ent.label_ == "PER":
                    names.add(ent.text)

        return list(names)

    def check_grammar(self, text: str) -> List[Dict]:
        """Kiểm tra ngữ pháp cơ bản trước khi gửi đến LanguageTool API."""
        issues = []
        
        # Kiểm tra subject-verb agreement
        patterns = [
            (r'\b(is|are|am|was|were)\b', self._check_subject_verb_agreement),
            (r'\b(has|have|had)\b', self._check_subject_verb_agreement),
        ]
        
        for pattern, checker_func in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if issue := checker_func(text, match):
                    issues.append(issue)
        
        return issues

    def _check_subject_verb_agreement(self, text: str, verb_match) -> Optional[Dict]:
        """Kiểm tra sự phù hợp giữa chủ ngữ và động từ."""
        verb = verb_match.group()
        start_pos = verb_match.start()
        
        # Lấy chủ ngữ trước động từ
        before_verb = text[:start_pos].strip()
        words_before = before_verb.split()
        
        if not words_before:
            return None
            
        subject = words_before[-1].lower()
        
        # Kiểm tra số ít/số nhiều
        if subject in {'he', 'she', 'it'} and verb in {'are', 'were', 'have'}:
            return {
                'message': f"Subject-verb agreement error: '{subject}' requires singular verb form",
                'context': text[max(0, start_pos-20):min(len(text), start_pos+20)],
                'suggestions': [self._get_correct_verb(verb)],
                'rule': "Subject-verb agreement"
            }
        
        # Kiểm tra tên riêng
        potential_names = self.extract_potential_names(text)
        if words_before[-1] in potential_names and verb in {'are', 'were', 'have'}:
            return {
                'message': f"Subject-verb agreement error: The name requires singular verb form",
                'context': text[max(0, start_pos-20):min(len(text), start_pos+20)],
                'suggestions': [self._get_correct_verb(verb)],
                'rule': "Subject-verb agreement"
            }
            
        return None

    def _get_correct_verb(self, incorrect_verb: str) -> str:
        """Lấy dạng đúng của động từ."""
        corrections = {
            'are': 'is', 'is': 'are',
            'was': 'were', 'were': 'was',
            'has': 'have', 'have': 'has'
        }
        return corrections.get(incorrect_verb, incorrect_verb)

    def check_text(self, text: str) -> List[Dict]:
        # Trích xuất tên để bỏ qua
        ignore_words = self.extract_potential_names(text)
        
        # Kiểm tra ngữ pháp local
        grammar_issues = self.check_grammar(text)
        
        # Chuẩn bị data cho LanguageTool API
        data = {
            'text': text,
            'language': self.language,
            'disabledCategories': 'NAMES',
            'enabledOnly': 'false'
        }
        
        if ignore_words:
            data['ignoredWords'] = ','.join(ignore_words)

        try:
            response = requests.post(
                url=self.api_url,
                data=data,
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                timeout=10
            )
            response.raise_for_status()
            api_results = response.json()
            
            # Kết hợp kết quả kiểm tra local và API
            all_issues = grammar_issues + [
                {
                    'message': match['message'],
                    'context': match['context']['text'],
                    'suggestions': [fix['value'] for fix in match.get('replacements', [])[:3]],
                    'rule': match['rule'].get("description", "Unknown rule")
                }
                for match in api_results.get('matches', [])
                if not any(word in match['context']['text'] for word in ignore_words)
            ]
            
            # Loại bỏ trùng lặp
            seen = set()
            unique_issues = []
            for issue in all_issues:
                key = (issue['message'], issue['context'])
                if key not in seen:
                    seen.add(key)
                    unique_issues.append(issue)
            
            return unique_issues

        except requests.exceptions.RequestException as e:
            logger.error(f"LanguageTool API error: {str(e)}")
            raise GrammarCheckerException(f"LanguageTool API error: {str(e)}")
# SpellChecker Implementation
class SpellingChecker(BaseGrammarChecker):
    def __init__(self):
        self.spell = SpellChecker()

    def check_text(self, text: str) -> List[Dict]:
        words = text.split()
        word_positions = {}
        
        for index, word in enumerate(words):
            clean_word = word.strip('.,!?:;()[]{}""''')
            word_positions.setdefault(clean_word, []).append(index)
        
        misspelled = self.spell.unknown([word.strip('.,!?:;()[]{}""''') for word in words])
        
        return [
            {
                'word': word,
                'suggestions': list(self.spell.candidates(word)),
                'positions': word_positions.get(word, [])
            }
            for word in misspelled
        ]

# Gramformer Implementation
class GramformerChecker(BaseGrammarChecker):
    def __init__(self, use_gpu: bool = False):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.gf = self._setup_gramformer()

    def _setup_gramformer(self) -> Optional[Gramformer]:
        try:
            return Gramformer(models=1, use_gpu=True)
        except Exception as e:
            logger.error(f"Gramformer initialization error: {str(e)}")
            return None

    def check_text(self, text: str) -> Dict:
        if self.gf is None:
            raise GrammarCheckerException("Gramformer not properly initialized")

        try:
            corrections = self.gf.correct(text, max_candidates=2)
            suggestions = []

            for correction in corrections:
                if correction != text:
                    suggestions.extend(self._get_differences(text, correction))

            return {
                'original': text,
                'corrections': list(corrections),
                'suggestions': suggestions
            }
        except Exception as e:
            logger.error(f"Gramformer check error: {str(e)}")
            raise GrammarCheckerException(f"Grammar check failed: {str(e)}")

    def _get_differences(self, original: str, corrected: str) -> List[str]:
        return [
            f"Replace '{orig}' with '{corr}'"
            for orig, corr in zip(original.split(), corrected.split())
            if orig != corr
        ]

app = Flask(__name__)

checkers = {
    'languagetool': LanguageToolChecker(),
    'spellchecker': SpellingChecker(),
    'gramformer': GramformerChecker()
}

@app.errorhandler(Exception)
def handle_error(error):
    if isinstance(error, ValidationError):
        return jsonify({'error': str(error)}), 400
    if isinstance(error, GrammarCheckerException):
        return jsonify({'error': str(error)}), 500
    
    logger.error(f"Unexpected error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

@app.route('/check/<checker_type>', methods=['POST'])
@validate_request
def check_text(checker_type):
    if checker_type not in checkers:
        raise ValidationError(f"Invalid checker type. Available types: {', '.join(checkers.keys())}")

    text = request.json['text']
    checker = checkers[checker_type]
    
    try:
        result = checker.check_text(text)
        return jsonify(CorrectionResponse(
            success=True,
            data=result,
            total_issues=len(result) if isinstance(result, list) else len(result.get('corrections', []))
        ).__dict__)
    except Exception as e:
        logger.error(f"Error in {checker_type} checker: {str(e)}")
        raise GrammarCheckerException(f"Error in {checker_type} checker: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)