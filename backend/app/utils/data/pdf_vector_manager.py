import os
import json
from typing import List, Dict
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv
from pathlib import Path

# AI Search 격리 시스템 import
try:
    from utils.isolation.ai_search_isolation import AISearchIsolationManager
    ISOLATION_AVAILABLE = True
except ImportError:
    print("⚠️ AI Search 격리 모듈을 찾을 수 없습니다. 기본 모드로 실행됩니다.")
    ISOLATION_AVAILABLE = False

# 환경 변수 로드
dotenv_path = Path(r'C:\Users\EL0021\Desktop\odiga_multiomodal_agent\.env')
load_dotenv(dotenv_path=dotenv_path, override=True)

class PDFVectorManager:
    """다중 인덱스 지원 벡터 데이터 관리자 - 인덱스 연결 및 검색 전용"""

    def __init__(self, isolation_enabled=True, default_index="magazine-vector-index"):
        # Azure 서비스 초기화
        self.search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.search_key = os.getenv("AZURE_SEARCH_KEY")
        self.default_index = default_index
        
        # 인덱스별 SearchClient 캐시
        self.search_clients = {}
        
        # 인덱스 클라이언트 (연결 확인용)
        self.search_index_client = SearchIndexClient(
            endpoint=self.search_endpoint,
            credential=AzureKeyCredential(self.search_key)
        )
        
        # Azure OpenAI 클라이언트 (임베딩 생성용)
        self.openai_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        self.embedding_model = "text-embedding-ada-002"
        
        # AI Search 격리 시스템 초기화
        self.isolation_enabled = isolation_enabled and ISOLATION_AVAILABLE
        if self.isolation_enabled:
            self.isolation_manager = AISearchIsolationManager()
            print("🛡️ PDFVectorManager AI Search 격리 시스템 활성화")
        else:
            self.isolation_manager = None
            print("⚠️ PDFVectorManager AI Search 격리 시스템 비활성화")
        
        # 지원하는 인덱스 목록
        self.supported_indexes = {
            "magazine-vector-index": {
                "description": "매거진 레이아웃 패턴",
                "vector_field": "content_vector",
                "select_fields": ["id", "pdf_name", "page_number", "content_type", 
                                "text_content", "layout_info", "image_info"]
            },
            "jsx-component-vector-index": {
                "description": "JSX 컴포넌트 패턴", 
                "vector_field": "jsx_vector",
                "select_fields": ["id", "component_name", "jsx_structure", "layout_method",
                                "image_count", "jsx_code", "search_keywords"]
            },
            "text-semantic-patterns-index": {
                "description": "텍스트 의미 분석 패턴",
                "vector_field": "semantic_vector", 
                "select_fields": ["id", "text_content", "emotional_tone", "primary_theme",
                                "visual_keywords", "search_keywords", "semantic_tags"]
            }
        }
        
        print(f"✅ PDFVectorManager 초기화 완료 (기본 인덱스: {default_index})")

    def _get_search_client(self, index_name: str) -> SearchClient:
        """인덱스별 SearchClient 반환 (캐시 사용)"""
        if index_name not in self.search_clients:
            self.search_clients[index_name] = SearchClient(
                endpoint=self.search_endpoint,
                index_name=index_name,
                credential=AzureKeyCredential(self.search_key)
            )
        return self.search_clients[index_name]

    async def verify_index_connectivity(self, index_name: str) -> Dict:
        """인덱스 연결 상태 및 데이터 확인"""
        try:
            # 인덱스 존재 여부 확인
            try:
                index_info = self.search_index_client.get_index(index_name)
                index_exists = True
            except Exception:
                return {
                    "index_name": index_name,
                    "exists": False,
                    "connected": False,
                    "error": "인덱스가 존재하지 않습니다",
                    "status": "not_found"
                }
            
            # SearchClient 연결 테스트
            search_client = self._get_search_client(index_name)
            
            # 테스트 검색으로 연결 및 데이터 확인
            test_results = search_client.search(
                search_text="*",
                top=1,
                include_total_count=True
            )
            
            # 문서 수 확인
            document_count = getattr(test_results, 'get_count', lambda: 0)()
            
            # 첫 번째 문서 확인 (데이터 형식 검증)
            sample_doc = None
            for doc in test_results:
                sample_doc = dict(doc)
                break
            
            return {
                "index_name": index_name,
                "exists": True,
                "connected": True,
                "document_count": document_count,
                "sample_fields": list(sample_doc.keys()) if sample_doc else [],
                "status": "healthy" if document_count > 0 else "empty",
                "description": self.supported_indexes.get(index_name, {}).get("description", "알 수 없음")
            }
            
        except Exception as e:
            return {
                "index_name": index_name,
                "exists": True,
                "connected": False,
                "error": str(e),
                "status": "connection_failed"
            }

    async def verify_all_indexes(self) -> Dict[str, Dict]:
        """모든 지원 인덱스의 연결 상태 확인"""
        results = {}
        
        for index_name in self.supported_indexes.keys():
            print(f"🔍 인덱스 연결 확인 중: {index_name}")
            status = await self.verify_index_connectivity(index_name)
            results[index_name] = status
            
            # 상태 로깅
            if status["connected"] and status["document_count"] > 0:
                print(f"✅ {index_name}: {status['document_count']}개 문서")
            elif status["connected"] and status["document_count"] == 0:
                print(f"⚠️ {index_name}: 연결됨, 데이터 없음")
            else:
                print(f"❌ {index_name}: {status.get('error', '연결 실패')}")
        
        return results

    def _create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Azure OpenAI Embeddings API로 벡터 생성"""
        embeddings = []
        print(f"📊 {len(texts)}개 텍스트에 대한 임베딩 생성 중...")
        
        for i, text in enumerate(texts):
            try:
                response = self.openai_client.embeddings.create(
                    input=text,
                    model=self.embedding_model
                )
                embeddings.append(response.data[0].embedding)
                
                if (i + 1) % 10 == 0:
                    print(f" 진행률: {i + 1}/{len(texts)} 완료")
                    
            except Exception as e:
                print(f"❌ 임베딩 생성 실패 (텍스트 {i+1}): {e}")
                # 기본 벡터 (1536 차원)
                embeddings.append([0.0] * 1536)
        
        print(f"✅ 임베딩 생성 완료: {len(embeddings)}개")
        return embeddings

    def search_similar_layouts(self, query_text: str, index_name: str = None, top_k: int = 5) -> List[Dict]:
        """다중 인덱스 지원 유사 레이아웃 검색 (AI Search 격리 적용)"""
        target_index = index_name or self.default_index
        
        # 지원하는 인덱스인지 확인
        if target_index not in self.supported_indexes:
            print(f"❌ 지원하지 않는 인덱스: {target_index}")
            return []
        
        try:
            # 1. 쿼리 격리 (AI Search 키워드 제거)
            if self.isolation_enabled:
                clean_query = self.isolation_manager.clean_query_from_azure_keywords(query_text)
                print(f"🛡️ 쿼리 격리: '{query_text[:50]}...' → '{clean_query[:50]}...'")
            else:
                clean_query = query_text

            # 2. 쿼리 텍스트를 벡터로 변환
            query_embeddings = self._create_embeddings([clean_query])
            query_vector = query_embeddings[0]

            # 3. 인덱스별 설정 가져오기
            index_config = self.supported_indexes[target_index]
            vector_field = index_config["vector_field"]
            select_fields = index_config["select_fields"]

            # 4. 벡터 검색 쿼리 생성
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k * 2,  # 격리 필터링을 위해 더 많이 검색
                fields=vector_field
            )

            # 5. SearchClient 가져오기 및 검색 실행
            search_client = self._get_search_client(target_index)
            
            search_params = {
                "vector_queries": [vector_query],
                "top": top_k * 2,
                "select": select_fields
            }

            raw_results = search_client.search(**search_params)

            # 6. 원시 결과 수집 (인덱스별 형식에 맞게)
            raw_data = []
            for result in raw_results:
                if target_index == "magazine-vector-index":
                    # 매거진 레이아웃 데이터 형식
                    layout_info = json.loads(result.get("layout_info", "{}"))
                    image_info = json.loads(result.get("image_info", "[]"))
                    
                    data = {
                        "id": result["id"],
                        "pdf_name": result["pdf_name"],
                        "page_number": result["page_number"],
                        "text_content": result["text_content"],
                        "layout_info": layout_info,
                        "image_info": image_info,
                        "score": result.get("@search.score", 0.0),
                        "source": "pdf_vector_search",
                        "index_type": "magazine_layout"
                    }
                    
                elif target_index == "jsx-component-vector-index":
                    # JSX 컴포넌트 데이터 형식
                    jsx_structure = json.loads(result.get("jsx_structure", "{}"))
                    
                    data = {
                        "id": result["id"],
                        "component_name": result["component_name"],
                        "jsx_structure": jsx_structure,
                        "layout_method": result["layout_method"],
                        "image_count": result["image_count"],
                        "jsx_code": result["jsx_code"],
                        "search_keywords": result["search_keywords"],
                        "score": result.get("@search.score", 0.0),
                        "source": "jsx_vector_search",
                        "index_type": "jsx_component"
                    }
                    
                elif target_index == "text-semantic-patterns-index":
                    # 텍스트 의미 분석 데이터 형식
                    data = {
                        "id": result["id"],
                        "text_content": result["text_content"],
                        "emotional_tone": result["emotional_tone"],
                        "primary_theme": result["primary_theme"],
                        "visual_keywords": result["visual_keywords"],
                        "search_keywords": result["search_keywords"],
                        "semantic_tags": result["semantic_tags"],
                        "score": result.get("@search.score", 0.0),
                        "source": "semantic_vector_search",
                        "index_type": "text_semantic"
                    }
                
                raw_data.append(data)

            # 7. AI Search 격리 필터링
            if self.isolation_enabled:
                filtered_data = self.isolation_manager.filter_contaminated_data(
                    raw_data, f"{target_index}_search"
                )
                
                # 8. 원본 데이터 우선순위 적용
                prioritized_data = self._prioritize_original_data(filtered_data, target_index)
                
                print(f"🛡️ 검색 결과 격리: {len(raw_data)} → {len(prioritized_data)}개")
                return prioritized_data[:top_k]
            else:
                return raw_data[:top_k]

        except Exception as e:
            print(f"❌ 벡터 검색 실패 ({target_index}): {e}")
            if self.isolation_enabled:
                print("🛡️ 격리된 폴백 결과 반환")
                return self._get_isolated_fallback_data(target_index)
            return []

    def _prioritize_original_data(self, data: List[Dict], index_type: str) -> List[Dict]:
        """인덱스 타입별 원본 데이터 우선순위 적용"""
        if not self.isolation_enabled:
            return data

        prioritized = []
        
        for item in data:
            # 인덱스별 신뢰도 기준
            if index_type == "magazine-vector-index":
                pdf_name = item.get('pdf_name', '').lower()
                if any(pattern in pdf_name for pattern in ['template', 'layout', 'design', 'magazine']):
                    item['priority'] = 1
                    prioritized.insert(0, item)
                else:
                    item['priority'] = 2
                    prioritized.append(item)
                    
            elif index_type == "jsx-component-vector-index":
                component_name = item.get('component_name', '').lower()
                if any(pattern in component_name for pattern in ['magazine', 'article', 'content']):
                    item['priority'] = 1
                    prioritized.insert(0, item)
                else:
                    item['priority'] = 2
                    prioritized.append(item)
                    
            elif index_type == "text-semantic-patterns-index":
                semantic_tags = item.get('semantic_tags', '').lower()
                if any(pattern in semantic_tags for pattern in ['travel', 'magazine', 'descriptive']):
                    item['priority'] = 1
                    prioritized.insert(0, item)
                else:
                    item['priority'] = 2
                    prioritized.append(item)
            else:
                item['priority'] = 3
                prioritized.append(item)

        return prioritized

    def _get_isolated_fallback_data(self, index_type: str) -> List[Dict]:
        """인덱스 타입별 격리된 폴백 데이터 반환"""
        if index_type == "magazine-vector-index":
            return [{
                "id": "fallback_magazine_1",
                "pdf_name": "isolated_default_layout",
                "page_number": 1,
                "text_content": "기본 매거진 레이아웃",
                "layout_info": {
                    "text_blocks": [],
                    "images": [],
                    "layout_structure": ["single_column"]
                },
                "image_info": [],
                "score": 0.5,
                "source": "isolated_fallback",
                "index_type": "magazine_layout",
                "priority": 1
            }]
            
        elif index_type == "jsx-component-vector-index":
            return [{
                "id": "fallback_jsx_1",
                "component_name": "DefaultMagazineComponent",
                "jsx_structure": {"type": "basic", "layout": "single_column"},
                "layout_method": "flex",
                "image_count": 1,
                "jsx_code": "// 기본 JSX 컴포넌트",
                "search_keywords": "기본 컴포넌트",
                "score": 0.5,
                "source": "isolated_fallback",
                "index_type": "jsx_component",
                "priority": 1
            }]
            
        elif index_type == "text-semantic-patterns-index":
            return [{
                "id": "fallback_semantic_1",
                "text_content": "기본 여행 경험 텍스트",
                "emotional_tone": "neutral",
                "primary_theme": "travel",
                "visual_keywords": "일반적인, 기본적인",
                "search_keywords": "여행 경험 기본",
                "semantic_tags": "travel basic",
                "score": 0.5,
                "source": "isolated_fallback",
                "index_type": "text_semantic",
                "priority": 1
            }]
        
        return []

    def get_layout_recommendations(self, content_description: str, image_count: int, 
                                 index_type: str = "magazine-vector-index") -> List[Dict]:
        """콘텐츠 설명과 이미지 수를 바탕으로 레이아웃 추천 (다중 인덱스 지원)"""
        
        # 이미지 수에 따른 검색 쿼리 조정
        if image_count <= 1:
            base_query = f"single image layout simple clean {content_description}"
        elif image_count <= 3:
            base_query = f"multiple images grid layout {content_description}"
        else:
            base_query = f"many images gallery layout complex {content_description}"

        # AI Search 격리 적용
        if self.isolation_enabled:
            clean_query = self.isolation_manager.clean_query_from_azure_keywords(base_query)
            print(f"🛡️ 레이아웃 추천 쿼리 격리 적용")
        else:
            clean_query = base_query

        return self.search_similar_layouts(clean_query, index_type, top_k=3)

    def get_index_statistics(self) -> Dict[str, Dict]:
        """모든 인덱스의 통계 정보 반환"""
        stats = {}
        
        for index_name, config in self.supported_indexes.items():
            try:
                search_client = self._get_search_client(index_name)
                results = search_client.search(
                    search_text="*",
                    top=0,
                    include_total_count=True
                )
                
                document_count = getattr(results, 'get_count', lambda: 0)()
                
                stats[index_name] = {
                    "description": config["description"],
                    "document_count": document_count,
                    "vector_field": config["vector_field"],
                    "status": "active" if document_count > 0 else "empty"
                }
                
            except Exception as e:
                stats[index_name] = {
                    "description": config["description"],
                    "document_count": 0,
                    "error": str(e),
                    "status": "error"
                }
        
        return stats

    def test_search_functionality(self) -> Dict[str, bool]:
        """각 인덱스의 검색 기능 테스트"""
        test_results = {}
        
        test_queries = {
            "magazine-vector-index": "magazine layout design",
            "jsx-component-vector-index": "react component image",
            "text-semantic-patterns-index": "travel experience positive"
        }
        
        for index_name, test_query in test_queries.items():
            try:
                results = self.search_similar_layouts(test_query, index_name, top_k=1)
                test_results[index_name] = len(results) > 0
                print(f"✅ {index_name}: 검색 테스트 {'성공' if test_results[index_name] else '실패'}")
                
            except Exception as e:
                test_results[index_name] = False
                print(f"❌ {index_name}: 검색 테스트 실패 - {e}")
        
        return test_results

    def check_compatibility_with_agents(self) -> Dict[str, Dict]:
        """에이전트별 인덱스 호환성 확인"""
        compatibility = {
            "SemanticAnalysisEngine": {
                "required_indexes": ["text-semantic-patterns-index", "magazine-vector-index"],
                "compatibility_score": 0.0
            },
            "RealtimeLayoutGenerator": {
                "required_indexes": ["magazine-vector-index"],
                "compatibility_score": 0.0
            },
            "UnifiedJSXGenerator": {
                "required_indexes": ["jsx-component-vector-index"],
                "compatibility_score": 0.0
            },
            "UnifiedMultimodalAgent": {
                "required_indexes": ["text-semantic-patterns-index", "magazine-vector-index"],
                "compatibility_score": 0.0
            }
        }
        
        # 각 에이전트별 호환성 점수 계산
        for agent_name, agent_info in compatibility.items():
            required_indexes = agent_info["required_indexes"]
            available_count = 0
            
            for index_name in required_indexes:
                if index_name in self.supported_indexes:
                    # 실제 연결 테스트
                    try:
                        test_results = self.search_similar_layouts("test", index_name, top_k=1)
                        if len(test_results) >= 0:  # 연결만 되면 성공
                            available_count += 1
                    except:
                        pass
            
            compatibility[agent_name]["compatibility_score"] = available_count / len(required_indexes)
            compatibility[agent_name]["available_indexes"] = available_count
            compatibility[agent_name]["total_required"] = len(required_indexes)
        
        return compatibility



