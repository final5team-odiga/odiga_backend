import hashlib
import json

class CacheManager:
    """간단한 인메모리 캐시 관리자"""
    def __init__(self):
        self.cache = {}

    def _generate_key(self, data: any) -> str:
        """입력 데이터로부터 안정적인 캐시 키 생성"""
        if isinstance(data, (dict, list)):
            # dict, list는 정렬된 json 문자열로 변환하여 일관성 유지
            serializable_data = json.dumps(data, sort_keys=True)
        else:
            serializable_data = str(data)
        
        return hashlib.sha256(serializable_data.encode('utf-8')).hexdigest()

    def get(self, key_data: any):
        """캐시에서 데이터 조회"""
        key = self._generate_key(key_data)
        return self.cache.get(key)

    def set(self, key_data: any, value: any):
        """캐시에 데이터 저장"""
        key = self._generate_key(key_data)
        self.cache[key] = value

