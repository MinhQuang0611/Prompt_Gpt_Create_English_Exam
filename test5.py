from gramformer import Gramformer
import torch

def setup_gramformer(use_gpu=False):
    """Khởi tạo mô hình Gramformer"""
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    try:
        gf = Gramformer(models=1, use_gpu=use_gpu)  # 1 = sửa lỗi ngữ pháp
        return gf
    except Exception as e:
        print(f"Lỗi khi khởi tạo Gramformer: {str(e)}")
        return None

def check_grammar(text, gf):
    """
    Kiểm tra ngữ pháp và trả về các sửa đổi và gợi ý
    
    Tham số:
        text (str): Văn bản đầu vào cần kiểm tra
        gf: Instance của Gramformer
    
    Trả về:
        dict: Dictionary chứa văn bản gốc và các sửa đổi
    """
    if gf is None:
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
        
        # Lấy các sửa đổi ngữ pháp
        corrections = gf.correct(text, max_candidates=2)
        
        for correction in corrections:
            results['corrections'].append(correction)
            
            # Tạo các gợi ý cải thiện cụ thể
            if correction != text:
                differences = get_differences(text, correction)
                results['suggestions'].extend(differences)
        
        return results
    except Exception as e:
        return {
            'original': text,
            'corrections': [],
            'suggestions': [],
            'error': f'Lỗi khi kiểm tra ngữ pháp: {str(e)}'
        }

def get_differences(original, corrected):
    """
    So sánh văn bản gốc và văn bản đã sửa để tạo các gợi ý cụ thể
    """
    suggestions = []
    orig_words = original.split()
    corr_words = corrected.split()
    
    for i, (orig, corr) in enumerate(zip(orig_words, corr_words)):
        if orig != corr:
            suggestions.append(f"Nên thay '{orig}' bằng '{corr}'")
    
    return suggestions

def main():
    # Khởi tạo Gramformer
    print("Đang tải mô hình Gramformer...")
    gf = setup_gramformer()
    
    while True:
        # Lấy input từ người dùng
        text = input("\nNhập văn bản cần kiểm tra (hoặc 'quit' để thoát): ")
        
        if text.lower() == 'quit':
            break
            
        print("\nĐang kiểm tra văn bản:", text)
        results = check_grammar(text, gf)
        
        if 'error' in results:
            print("Lỗi:", results['error'])
            continue
            
        print("\nKết quả:")
        print("Văn bản gốc:", results['original'])
        print("\nCác sửa đổi:")
        for i, correction in enumerate(results['corrections'], 1):
            print(f"{i}. {correction}")
        
        print("\nCác gợi ý:")
        for suggestion in results['suggestions']:
            print("-", suggestion)

if __name__ == "__main__":
    main()