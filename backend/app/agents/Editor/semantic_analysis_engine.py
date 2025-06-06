import asyncio
import json
import numpy as np
import open_clip
import time
import torch
from typing import Dict, List, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from custom_llm import get_azure_llm
from utils.log.hybridlogging import get_hybrid_logger
from utils.isolation.ai_search_isolation import AISearchIsolationManager
from utils.data.pdf_vector_manager import PDFVectorManager
from utils.isolation.session_isolation import SessionAwareMixin
from utils.isolation.agent_communication_isolation import InterAgentCommunicationMixin
from agents.Editor.image_diversity_manager import ImageDiversityManager
from utils.log.logging_manager import LoggingManager
from db.magazine_db_utils import MagazineDBUtils

class SemanticAnalysisEngine(SessionAwareMixin, InterAgentCommunicationMixin):
    """의미적 분석 엔진 - 텍스트와 이미지의 의미적 연관성 분석 (AI Search 통합)"""
    
    def __init__(self):
        self.llm = get_azure_llm()
        self.logger = get_hybrid_logger(self.__class__.__name__)
        # AI Search 격리 시스템 추가
        self.isolation_manager = AISearchIsolationManager()
        # PDF 벡터 매니저 추가 (격리 활성화)
        self.vector_manager = PDFVectorManager(isolation_enabled=True)
        self.logging_manager = LoggingManager(self.logger)
        self.__init_session_awareness__()
        self.__init_inter_agent_communication__()
        self.image_diversity_manager = ImageDiversityManager(self.logger)

        self._setup_logging_system()

    def _setup_logging_system(self):
        """로그 저장 시스템 설정"""
        self.log_enabled = True
        self.response_counter = 0

    async def process_data(self, input_data):
        # 에이전트 작업 수행
        result = await self._do_work(input_data)
        
        # ✅ 응답 로그 저장
        await self.logging_manager.log_agent_response(
            agent_name=self.__class__.__name__,
            agent_role="에이전트 역할 설명",
            task_description="수행한 작업 설명",
            response_data=result,  # 실제 응답 데이터만
            metadata={"additional": "info"}
        )

    async def _log_semantic_analysis_response(self, analysis_result: Dict) -> str:
        """의미적 분석 결과 로그 저장 (BindingAgent 방식 적용)"""
        if not self.log_enabled:
            return "logging_disabled"
        
        try:
            response_data = {
                "agent_name": "SemanticAnalysisEngine",
                "analysis_type": "text_image_semantics",
                "text_sections": len(analysis_result.get("text_semantics", [])),
                "image_sections": len(analysis_result.get("image_semantics", [])),
                "mapping_confidence": analysis_result.get("analysis_metadata", {}).get("mapping_confidence", 0.0),
                "ai_search_enhanced": analysis_result.get("analysis_metadata", {}).get("ai_search_enhanced", False),
                "timestamp": time.time(),
                "session_id": self.current_session_id
            }
            
            response_id = f"SemanticAnalysis_{int(time.time() * 1000000)}"
            
            # self.store_result(response_data)  # 세션 저장 제거 - 사용되지 않음
            
            self.logger.info(f"📦 SemanticAnalysisEngine 응답 저장: {response_id}")
            return response_id
            
        except Exception as e:
            self.logger.error(f"로그 저장 실패: {e}")
            return "log_save_failed"
        
    async def analyze_text_image_semantics(self, magazine_content: Dict, image_analysis: List[Dict]) -> Dict:
        """의미적 텍스트-이미지 매칭 (구조 통일)"""
        
        try:
            # magazine_id가 있으면 Cosmos DB에서 최신 데이터 조회
            if "magazine_id" in magazine_content:
                magazine_data = await MagazineDBUtils.get_magazine_by_id(magazine_content["magazine_id"])
                if magazine_data:
                    magazine_content = magazine_data.get("content", magazine_content)
                    
                    # 이미지 분석 결과도 Cosmos DB에서 조회 (변경된 저장 방식 적용)
                    image_analysis = await MagazineDBUtils.get_images_by_magazine_id(magazine_content["magazine_id"])
            
            sections = magazine_content.get("sections", [])
            if not sections:
                return self._generate_clean_fallback_result(magazine_content, image_analysis)
            
            if not image_analysis:
                # ✅ 이미지가 없어도 텍스트 분석은 수행
                text_semantics = await self._extract_text_semantics_with_vector_search(magazine_content)
                return {
                    "text_semantics": text_semantics,
                    "semantic_mappings": self._create_text_only_mappings(text_semantics),
                    "analysis_metadata": {
                        "sections_processed": len(sections),
                        "images_processed": len(image_analysis) if image_analysis else 0,
                        "success": True,
                        "text_only_mode": True
                    }
                }
            
            # ✅ 정상적인 텍스트-이미지 분석
            # CLIP 모델 초기화 확인
            await self._ensure_clip_initialization()
            
            # 텍스트 의미 분석
            text_semantics = await self._extract_text_semantics_with_vector_search(magazine_content)
            
            # 이미지 의미 분석
            image_semantics = await self._extract_image_semantics_with_layout_patterns_batch(image_analysis)
            
            # 의미적 매칭 수행
            semantic_mappings = await self._perform_enhanced_semantic_matching(
                text_semantics, image_semantics, sections, image_analysis
            )
            
            # ✅ 통일된 구조로 반환
            result = {
                "text_semantics": text_semantics,
                "image_semantics": image_semantics,
                "semantic_mappings": semantic_mappings,
                "optimal_combinations": await self._generate_optimal_combinations_with_ai_search(semantic_mappings),
                "analysis_metadata": {
                    "sections_processed": len(sections),
                    "images_processed": len(image_analysis),
                    "text_sections_analyzed": len(text_semantics),
                    "image_sections_analyzed": len(image_semantics),
                    "mappings_created": len(semantic_mappings),
                    "clip_available": getattr(self, 'clip_available', False),
                    "success": True
                }
            }
            
            # 분석 결과를 매거진에 업데이트
            if "magazine_id" in magazine_content:
                await MagazineDBUtils.update_magazine_content(magazine_content["magazine_id"], {
                    "content": magazine_content,
                    "semantic_analysis": result,
                    "status": "analyzed"
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"의미적 분석 실패: {e}")
            return self._generate_clean_fallback_result(magazine_content, image_analysis)

    def _create_text_only_mappings(self, text_semantics: List[Dict]) -> List[Dict]:
        """텍스트만 있을 때의 매핑 생성"""
        mappings = []
        for text_section in text_semantics:
            mappings.append({
                "text_section_index": text_section["section_index"],
                "text_title": text_section["title"],
                "image_matches": [],  # 이미지 없음
                "text_only": True
            })
        return mappings

    async def _ensure_clip_initialization(self):
        """CLIP 모델 초기화 보장"""
        if not hasattr(self, 'clip_available'):
            try:
                import torch
                import open_clip
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32', pretrained='laion2b_s34b_b79k', device=self.device
                )
                self.clip_model.eval()
                self.clip_available = True
                self.logger.info("✅ CLIP 모델 초기화 성공")
            except Exception as e:
                self.logger.warning(f"⚠️ CLIP 모델 초기화 실패, 품질 기반 매칭 사용: {e}")
                self.clip_available = False

    async def _perform_enhanced_semantic_matching(self, text_semantics: List[Dict], 
                                                image_semantics: List[Dict],
                                                sections: List[Dict], 
                                                image_analysis: List[Dict]) -> List[Dict]:
        """강화된 의미적 매칭 (빈 결과 방지)"""
        
        if not text_semantics:
            self.logger.warning("텍스트 의미 분석 결과 없음. 빈 매핑 반환.")
            return [] # 빈 리스트 반환
        
        # CLIP 기반 매칭 시도
        if self.clip_available and image_analysis and image_semantics: # image_semantics도 확인
            try:
                # image_analysis 대신 image_semantics를 전달하여 분석된 정보를 활용하도록 고려 가능
                # 여기서는 image_analysis (원본 이미지 정보 리스트)를 계속 사용
                clip_mappings = await self._perform_clip_based_matching(text_semantics, image_analysis, sections)
                if clip_mappings: # CLIP 매칭 결과가 있다면 반환
                    return clip_mappings
                else:
                    self.logger.warning("CLIP 매칭 결과가 비었음. 키워드 기반으로 폴백 시도.")
            except Exception as e:
                self.logger.error(f"CLIP 매칭 중 심각한 오류 발생: {e}. 키워드 기반으로 폴백 시도.")
        
        # 키워드 기반 매칭 (CLIP 실패 또는 결과 없음 시)
        # image_analysis 대신 image_semantics를 전달하는 것을 고려할 수 있으나, 현재 구조 유지
        if image_analysis:
            self.logger.info("키워드 기반 매칭 수행.")
            keyword_mappings = await self._perform_keyword_based_matching(text_semantics, image_analysis)
            if keyword_mappings:
                return keyword_mappings
            else:
                self.logger.warning("키워드 기반 매칭 결과도 비었음. 텍스트 전용 매핑으로 폴백.")
        
        # 이미지 분석 결과가 아예 없거나 모든 매칭 시도가 실패한 경우, 텍스트 전용 매핑 생성
        self.logger.info("텍스트 전용 매핑 생성.")
        return self. _create_text_only_mappings(text_semantics)

    async def _perform_clip_based_matching(self, text_semantics: List[Dict], 
                                            image_analysis: List[Dict], # 전체 이미지 분석 결과 (URL 등 포함)
                                            sections: List[Dict]) -> List[Dict]: # 원본 텍스트 섹션 (참고용)
        """CLIP 기반 의미적 매칭. 각 텍스트 섹션과 전체 이미지 풀 간의 유사도를 계산하여 매핑."""
        
        if not self.clip_available:
            self.logger.warning("CLIP model not available for _perform_clip_based_matching. Returning empty list.")
            return []

        # 1. 텍스트 임베딩 생성
        section_texts_for_embedding = []
        valid_text_indices = [] # 임베딩 생성이 가능한 텍스트 섹션의 원본 인덱스
        for i, ts_item in enumerate(text_semantics):
            title = ts_item.get("title", "")
            # content_preview가 있으면 사용, 없으면 원본 sections에서 가져오기 시도
            content_preview = ts_item.get("content_preview", "")
            if not content_preview and i < len(sections):
                content_preview = sections[i].get("content", "")[:200]
            
            text_for_embedding = (title + " " + content_preview).strip()[:500] # 임베딩할 텍스트 길이 제한 (예: 500자)
            if not text_for_embedding: 
                 text_for_embedding = "empty section" # 내용이 전혀 없으면 기본값 사용
            section_texts_for_embedding.append(text_for_embedding)
            valid_text_indices.append(ts_item.get("section_index", i)) # 원본 section_index 사용

        if not section_texts_for_embedding:
             self.logger.warning("No text content available in text_semantics for CLIP matching. Returning empty list.")
             return []

        text_embeddings_np = await self._generate_clip_text_embeddings(section_texts_for_embedding)
        if text_embeddings_np.size == 0 or text_embeddings_np.shape[0] != len(section_texts_for_embedding):
            self.logger.error("Failed to generate text embeddings or shape mismatch. Returning empty list.")
            return []

        # 2. 이미지 임베딩 생성
        if not image_analysis: 
            self.logger.info("No images provided for CLIP matching. Returning text-only mappings based on input text_semantics.")
            # 이 경우, semantic_mappings에 image_matches를 빈 리스트로 채워서 반환할 수 있음
            # 또는 _create_text_only_mappings와 유사한 구조로 여기서 직접 생성
            # 여기서는 일단 빈 리스트 반환하여 _perform_enhanced_semantic_matching에서 처리하도록 함
            return [] 

        self.logger.info(f"Generating image embeddings for {len(image_analysis)} images for CLIP matching.")
        image_embeddings_np = await self._generate_clip_image_embeddings_from_data(image_analysis)
        
        if image_embeddings_np.size == 0 or image_embeddings_np.shape[0] != len(image_analysis):
            self.logger.warning("Image embeddings array is empty or shape mismatch for CLIP matching. Returning empty list.")
            return []

        # 3. 임베딩 차원 확인 및 2D로 변환 (필요시)
        if text_embeddings_np.ndim == 1: text_embeddings_np = text_embeddings_np.reshape(1, -1)
        if image_embeddings_np.ndim == 1: image_embeddings_np = image_embeddings_np.reshape(1, -1)
        
        if text_embeddings_np.shape[1] != image_embeddings_np.shape[1]:
            self.logger.error(
                f"Embedding dimension mismatch: Texts {text_embeddings_np.shape[1]}, Images {image_embeddings_np.shape[1]}. Cannot compute similarity."
            )
            return []

        # 4. 코사인 유사도 계산 (모든 텍스트 섹션 vs 모든 이미지)
        try:
            # similarity_matrix[i, j]는 i번째 텍스트와 j번째 이미지 간의 유사도
            similarity_matrix = cosine_similarity(text_embeddings_np, image_embeddings_np) 
        except ValueError as ve:
            self.logger.error(f"Error calculating cosine similarity: {ve}. Check embedding dimensions and content. Returning empty list.")
            return []

        # 5. semantic_mappings 생성
        semantic_mappings_result = []
        for i, original_text_section_info in enumerate(text_semantics): # 원본 text_semantics 리스트를 순회
            # text_embeddings_np[i]에 해당하는 텍스트 섹션은 section_texts_for_embedding[i]
            # original_text_section_info는 이 텍스트 섹션의 원본 메타데이터 (title, section_index 등)
            
            # 현재 텍스트 섹션과 모든 이미지 간의 유사도 점수 벡터
            image_scores_for_this_text_vector = similarity_matrix[i] 
            
            image_matches_for_section = []
            for j, image_data_from_analysis in enumerate(image_analysis): # 전체 image_analysis 리스트 순회
                similarity_score = float(image_scores_for_this_text_vector[j])
                
                # image_data_from_analysis는 이미 image_name, image_url 등의 정보를 포함
                # 여기에 CLIP 유사도 점수 및 원본 이미지 리스트에서의 인덱스 추가
                matched_image_info = {
                    **image_data_from_analysis, 
                    "similarity_score": round(similarity_score, 4), # 소수점 4자리
                    "image_index": j # image_analysis 리스트에서의 인덱스 (키 이름 변경)
                }
                image_matches_for_section.append(matched_image_info)

            # 각 섹션별 이미지 후보들을 유사도 높은 순으로 정렬
            image_matches_for_section.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            # (선택적) 상위 N개 이미지 필터링 (예: 상위 10개 또는 특정 임계값 이상)
            # top_n_filter = 15
            # image_matches_for_section = image_matches_for_section[:top_n_filter]

            semantic_mappings_result.append({
                "text_section_index": original_text_section_info.get("section_index"), # 원본 text_semantics의 section_index 사용
                "text_title": original_text_section_info.get("title"), # 원본 title 사용
                "image_matches": image_matches_for_section # 이 텍스트 섹션에 대한 모든 이미지 후보군 (정렬됨)
            })
        
        return semantic_mappings_result

    async def _perform_keyword_based_matching(self, text_semantics: List[Dict], 
                                                image_analysis: List[Dict]) -> List[Dict]:
        """키워드 기반 의미적 매칭 (CLIP 대안)"""
        
        semantic_mappings = []
        
        for text_section in text_semantics:
            # 텍스트에서 키워드 추출
            text_keywords = self._extract_keywords_from_text(text_section)
            
            # 이미지와 키워드 매칭
            image_matches = []
            for i, img in enumerate(image_analysis):
                match_score = self._calculate_keyword_similarity(text_keywords, img)
                
                if match_score > 0.1:  # 최소 임계값
                    match = img.copy()
                    match["image_index"] = i
                    match["similarity_score"] = match_score
                    image_matches.append(match)
            
            # 점수 순으로 정렬하여 상위 3개 선택
            image_matches.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            semantic_mappings.append({
                "text_section_index": text_section["section_index"],
                "text_title": text_section["title"],
                "image_matches": image_matches[:3]
            })
        
        return semantic_mappings

    def _extract_keywords_from_text(self, text_section: Dict) -> List[str]:
        """텍스트에서 키워드 추출"""
        keywords = []
        
        # 제목에서 키워드 추출
        title = text_section.get("title", "")
        keywords.extend(title.split())
        
        # 의미 분석 결과에서 키워드 추출
        semantic_analysis = text_section.get("semantic_analysis", {})
        
        if isinstance(semantic_analysis, dict):
            # 주요 주제
            main_topics = semantic_analysis.get("주요_주제", [])
            if isinstance(main_topics, list):
                keywords.extend(main_topics)
            
            # 시각적 키워드
            visual_keywords = semantic_analysis.get("시각적_키워드", [])
            if isinstance(visual_keywords, list):
                keywords.extend(visual_keywords)
            
            # 문화적 요소
            cultural_elements = semantic_analysis.get("문화적_요소", [])
            if isinstance(cultural_elements, list):
                keywords.extend(cultural_elements)
        
        return [k.strip() for k in keywords if k.strip()]

    def _calculate_keyword_similarity(self, text_keywords: List[str], image: Dict) -> float:
        """키워드 기반 유사도 계산"""
        if not text_keywords:
            return 0.0
        
        # 이미지 정보에서 키워드 추출
        image_text = ""
        image_text += image.get("location", "") + " "
        image_text += image.get("description", "") + " "
        image_text += image.get("city", "") + " "
        image_text += image.get("country", "") + " "
        
        image_keywords = image_text.lower().split()
        
        # 공통 키워드 계산
        text_keywords_lower = [k.lower() for k in text_keywords]
        common_keywords = set(text_keywords_lower) & set(image_keywords)
        
        if not text_keywords_lower:
            return 0.0
        
        similarity = len(common_keywords) / len(text_keywords_lower)
        return min(similarity, 1.0)

    async def _generate_clip_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """텍스트 리스트에 대한 CLIP 텍스트 임베딩 생성"""
        if not self.clip_available or not texts:
            self.logger.warning("CLIP model not available or no texts provided for text embedding.")
            # 적절한 차원의 0 벡터 또는 빈 배열 반환
            output_dim = 512 # 기본값
            if hasattr(self, 'clip_model') and hasattr(self.clip_model, 'text_projection') and self.clip_model.text_projection is not None:
                 # 모델에서 실제 output 차원을 가져오려고 시도 (예시, 실제 모델 구조에 따라 다름)
                 if isinstance(self.clip_model.text_projection, torch.Tensor): # Check if it's a Tensor
                    output_dim = self.clip_model.text_projection.shape[-1]
                 elif isinstance(self.clip_model.text_projection, torch.nn.Parameter):
                    output_dim = self.clip_model.text_projection.shape[-1]
                 # 다른 가능한 속성들 (예: self.clip_model.transformer.width)
                 elif hasattr(self.clip_model, 'transformer') and hasattr(self.clip_model.transformer, 'width'):
                     output_dim = self.clip_model.transformer.width


            return np.zeros((len(texts), output_dim), dtype=np.float32) if texts else np.array([])

        try:
            with torch.no_grad():
                    # `open_clip.tokenize`는 텍스트 리스트를 직접 받음
                text_tokens = open_clip.tokenize(texts).to(self.device)
                text_features = self.clip_model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                return text_features.cpu().numpy().astype(np.float32)
        except Exception as e:
            self.logger.error(f"Error generating CLIP text embeddings: {e}")
            output_dim = 512 # 기본값
            if hasattr(self, 'clip_model') and hasattr(self.clip_model, 'text_projection') and self.clip_model.text_projection is not None:
                if isinstance(self.clip_model.text_projection, torch.Tensor): # Check if it's a Tensor
                    output_dim = self.clip_model.text_projection.shape[-1]
                elif isinstance(self.clip_model.text_projection, torch.nn.Parameter):
                    output_dim = self.clip_model.text_projection.shape[-1]
                elif hasattr(self.clip_model, 'transformer') and hasattr(self.clip_model.transformer, 'width'):
                    output_dim = self.clip_model.transformer.width

            return np.zeros((len(texts), output_dim), dtype=np.float32)


    async def _generate_clip_image_embeddings_from_data(self, images: List[Dict]) -> np.ndarray:
        """
        주어진 이미지 데이터 리스트에서 CLIP 이미지 임베딩을 생성.
        캐시된 임베딩이 있으면 사용하고, 없으면 새로 계산.
        개별 이미지 로드/처리 실패 시 해당 이미지에 대해서는 0 벡터를 사용.
        """
        if not self.clip_available or not hasattr(self, 'clip_model') or not hasattr(self, 'clip_preprocess'):
            self.logger.warning("CLIP model not available for generating image embeddings.")
            output_dim = 512 # Fallback dimension if model is not fully available
            if hasattr(self, 'clip_model') and hasattr(self.clip_model, 'visual') and hasattr(self.clip_model.visual, 'output_dim'):
                output_dim = self.clip_model.visual.output_dim
            return np.zeros((len(images), output_dim), dtype=np.float32)

        all_embeddings = [None] * len(images) # Initialize with placeholders
        
        # Attempt to get output dimension from the model
        try:
            output_dim = self.clip_model.visual.output_dim
        except AttributeError:
            self.logger.error("Cannot determine CLIP model output dimension. Using fallback 512.")
            output_dim = 512 # Fallback dimension
            # If we can't get output_dim, it's a critical model issue, return all zeros
            return np.zeros((len(images), output_dim), dtype=np.float32)

        pil_images_to_process = []
        indices_for_pil_images = [] # Tracks original indices of images that will be processed by CLIP

        for idx, image_data in enumerate(images):
            image_url = image_data.get("image_url")
            if not image_url:
                self.logger.warning(f"Image at index {idx} (name: {image_data.get('image_name', 'N/A')}) has no URL. Using zero vector.")
                all_embeddings[idx] = np.zeros(output_dim, dtype=np.float32)
                continue

            # TODO: Implement caching mechanism if needed
            # cached_embedding = self.vector_manager.get_cached_image_embedding(image_url)
            # if cached_embedding is not None:
            #     all_embeddings[idx] = cached_embedding
            #     continue
            
            # If not cached, prepare for CLIP processing
            try:
                # Lazily import requests and PIL here if they are not commonly used elsewhere in this class
                from PIL import Image
                import requests
                from io import BytesIO

                response = requests.get(image_url, timeout=10)
                response.raise_for_status() # Raise an exception for bad status codes
                pil_img = Image.open(BytesIO(response.content)).convert("RGB")
                pil_images_to_process.append(pil_img)
                indices_for_pil_images.append(idx)
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Failed to download image from URL {image_url} (index {idx}): {e}. Using zero vector.")
                all_embeddings[idx] = np.zeros(output_dim, dtype=np.float32)
            except Image.UnidentifiedImageError: # PIL anead PIL.Image.UnidentifiedImageError
                self.logger.error(f"Cannot identify image file from URL {image_url} (index {idx}). It might be corrupted or not an image. Using zero vector.")
                all_embeddings[idx] = np.zeros(output_dim, dtype=np.float32)
            except Exception as e:
                self.logger.error(f"An unexpected error occurred while loading or preprocessing image {image_url} (index {idx}): {e}. Using zero vector.")
                all_embeddings[idx] = np.zeros(output_dim, dtype=np.float32)

        if pil_images_to_process: # If there are any images to process with CLIP
            try:
                image_inputs = torch.stack([self.clip_preprocess(img) for img in pil_images_to_process]).to(self.device)
                
                with torch.no_grad(), torch.cuda.amp.autocast():
                    image_features = self.clip_model.encode_image(image_inputs)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                
                batch_embeddings_np = image_features.cpu().numpy().astype(np.float32)

                # Place computed embeddings into the correct original positions
                for i, original_idx in enumerate(indices_for_pil_images):
                    all_embeddings[original_idx] = batch_embeddings_np[i]
            
            except Exception as e:
                self.logger.error(f"Error during CLIP model image encoding batch: {e}. Using zero vectors for these images.")
                for original_idx in indices_for_pil_images: # For images that were intended for this batch
                    all_embeddings[original_idx] = np.zeros(output_dim, dtype=np.float32)
        
        # Ensure all placeholders are filled (should be, due to error handling above)
        for idx_fill in range(len(all_embeddings)): # Renamed idx to idx_fill to avoid conflict
            if all_embeddings[idx_fill] is None: # Should not happen if logic is correct
                self.logger.error(f"Embedding for image at index {idx_fill} was unexpectedly None post-processing. Filling with zero vector.")
                all_embeddings[idx_fill] = np.zeros(output_dim, dtype=np.float32)
        
        return np.array(all_embeddings).astype(np.float32)


    async def _generate_optimal_combinations_with_ai_search(self, semantic_mappings: List[Dict]) -> List[Dict]:
        """
        의미적 분석 결과를 기반으로 섹션별 최적 이미지 조합 생성 (중복 없이)
        """
        optimal_combinations = []
        used_images = set()
        for mapping in semantic_mappings:
            section_index = mapping["text_section_index"]
            section_title = mapping["text_title"]
            best_images = []
            for image_match in mapping["image_matches"]:
                if image_match["image_index"] not in used_images:
                    best_images.append(image_match)
                    used_images.add(image_match["image_index"])
            optimal_combinations.append({
                "section_index": section_index,
                "section_title": section_title,
                "assigned_images": best_images,
                "total_similarity_score": sum(img["similarity_score"] for img in best_images),
                "ai_search_enhanced": True,
                "optimization_notes": f"{len(best_images)}개 이미지 할당됨 (중복 없이 의미적 매칭)"
            })
        return optimal_combinations
    
    async def _extract_text_semantics_with_vector_search(self, content: Dict) -> List[Dict]:
        """AI Search 벡터 검색을 활용한 텍스트 의미 추출"""
        
        text_sections = []
        
        if isinstance(content, dict) and "sections" in content:
            for i, section in enumerate(content["sections"]):
                # 섹션별 오염 검사
                if self.isolation_manager.is_contaminated(section, f"text_section_{i}"):
                    self.logger.warning(f"텍스트 섹션 {i}에서 오염 감지, 원본 데이터 사용")
                    section = self.isolation_manager.restore_original_content(section)
                
                # AI Search 벡터 검색으로 유사한 매거진 패턴 찾기
                section_content = section.get("content", "")
                similar_patterns = await self._search_similar_text_patterns(section_content)
                
                semantic_info = await self._analyze_text_section_with_patterns(section, i, similar_patterns)
                text_sections.append(semantic_info)
        
        return text_sections
    
    async def _search_similar_text_patterns(self, section_content: str) -> List[Dict]:
        """AI Search에서 유사한 텍스트 패턴 검색"""
        
        try:
            # AI Search 키워드 필터링
            clean_query = self.isolation_manager.clean_query_from_azure_keywords(section_content[:300])
            
            # 벡터 검색 실행 (텍스트 패턴 중심)
            similar_patterns = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.vector_manager.search_similar_layouts(
                    clean_query, "text-semantic-patterns-index", top_k=8
                )
            )
            
            # 격리된 패턴만 반환
            isolated_patterns = self.isolation_manager.filter_contaminated_data(
                similar_patterns, "text_patterns"
            )
            
            self.logger.debug(f"텍스트 패턴 검색: {len(similar_patterns)} → {len(isolated_patterns)}개")
            return isolated_patterns
            
        except Exception as e:
            self.logger.error(f"텍스트 패턴 검색 실패: {e}")
            return []
    
    async def _analyze_text_section_with_patterns(self, section: Dict, index: int, patterns: List[Dict]) -> Dict:
        """AI Search 패턴을 참조한 텍스트 섹션 분석"""
        
        section_content = section.get("content", "")
        title = section.get("title", "")
        
        # AI Search 키워드 필터링
        filtered_content = self.isolation_manager.clean_query_from_azure_keywords(section_content)
        filtered_title = self.isolation_manager.clean_query_from_azure_keywords(title)
        
        # 패턴 기반 분석 프롬프트 생성
        pattern_context = ""
        if patterns:
            pattern_info = []
            for pattern in patterns[:3]:  # 상위 3개 패턴만 사용
                pattern_info.append({
                    "글_형태": pattern.get("text_structure", "일반형"),
                    "문장_길이": pattern.get("sentence_length", "중간"),
                    "글_맺음": pattern.get("conclusion_style", "자연스러운"),
                    "섹션_구조": pattern.get("section_format", "제목-본문")
                })
            pattern_context = f"참조 패턴: {json.dumps(pattern_info, ensure_ascii=False)}"
        
        analysis_prompt = f"""
다음 텍스트 섹션의 의미적 요소를 분석하세요:

제목: {filtered_title}
내용: {filtered_content}

{pattern_context}

분석 항목:
1. 주요 주제 (구체적 키워드 추출)
2. 감정적 톤 (긍정적, 중성적, 성찰적 등)
3. 시각적 연관 키워드 (색상, 풍경, 건물, 사람 등)
4. 계절/시간대 정보
5. 문화적 요소
6. 글의 형태 (서술형, 대화형, 설명형 등)
7. 문장 길이 특성 (짧은/중간/긴 문장 비율)

반드시 JSON 형식으로 출력하세요.
"""
        
        try:
            response = await self.llm.ainvoke(analysis_prompt)
            
            #  강화된 응답 처리
            if not response or not response.strip():
                self.logger.warning(f"텍스트 섹션 {index} 분석에서 빈 응답 수신")
                return self._get_clean_section_fallback(index, title, section_content)
            
            #  응답 길이 체크
            if len(response.strip()) < 5:
                self.logger.warning(f"텍스트 섹션 {index} 응답이 너무 짧음: {response}")
                return self._get_clean_section_fallback(index, title, section_content)
            
            #  강화된 JSON 추출
            cleaned_response = self._extract_json_from_response(response.strip())
            
            #  JSON 검증 및 수정
            validated_json = self._validate_and_fix_json(cleaned_response)
            
            # JSON 파싱 시도
            try:
                analysis_result = json.loads(validated_json)
            except json.JSONDecodeError as json_error:
                self.logger.error(f"텍스트 섹션 {index} JSON 파싱 최종 실패: {json_error}")
                self.logger.debug(f"원본 응답: {response[:200]}...")
                self.logger.debug(f"정제된 응답: {cleaned_response[:200]}...")
                self.logger.debug(f"검증된 JSON: {validated_json[:200]}...")
                
                #  최후의 수단: 정규식으로 키워드 추출
                fallback_result = self._extract_keywords_with_regex(response)
                if fallback_result:
                    analysis_result = fallback_result
                else:
                    return self._get_clean_section_fallback(index, title, section_content)
            
            # 분석 결과 오염 검사
            if self.isolation_manager.is_contaminated(analysis_result, f"analysis_result_{index}"):
                self.logger.warning(f"분석 결과 {index}에서 오염 감지, 기본값 사용")
                analysis_result = self._get_clean_analysis_fallback()
            
            return {
                "section_index": index,
                "title": title,
                "content_preview": section_content[:200],
                "semantic_analysis": analysis_result,
                "confidence_score": 0.8,
                "ai_search_patterns": len(patterns),
                "isolation_metadata": {
                    "patterns_referenced": len(patterns),
                    "contamination_detected": False,
                    "original_preserved": True
                }
            }
            
        except Exception as e:
            self.logger.error(f"텍스트 섹션 {index} 분석 실패: {e}")
            return self._get_clean_section_fallback(index, title, section_content)
    
    def _extract_keywords_with_regex(self, response: str) -> Dict:
        """정규식을 사용한 키워드 추출 (최후의 수단)"""
        
        import re
        
        result = {
            "주요_주제": [],
            "감정적_톤": "중성적",
            "시각적_키워드": [],
            "계절_시간": "알 수 없음",
            "문화적_요소": [],
            "글의_형태": "서술형",
            "문장_길이_특성": "중간"
        }
        
        try:
            # 주요 주제 추출
            topic_patterns = [
                r'주요[_\s]*주제[:\s]*([^\n\r,]+)',
                r'키워드[:\s]*([^\n\r,]+)',
                r'여행|문화|예술'
            ]
            
            for pattern in topic_patterns:
                matches = re.findall(pattern, response, re.IGNORECASE)
                if matches:
                    result["주요_주제"].extend([m.strip() for m in matches if m.strip()])
            
            # 감정적 톤 추출
            tone_patterns = r'(긍정적|부정적|중성적|성찰적|감성적|낭만적)'
            tone_match = re.search(tone_patterns, response, re.IGNORECASE)
            if tone_match:
                result["감정적_톤"] = tone_match.group(1)
            
            # 시각적 키워드 추출
            visual_patterns = r'(색상|풍경|건물|바다|하늘|거리|광장|다리)'
            visual_matches = re.findall(visual_patterns, response, re.IGNORECASE)
            if visual_matches:
                result["시각적_키워드"] = visual_matches
            
            self.logger.info(f"정규식으로 키워드 추출 성공: {len(result['주요_주제'])}개 주제")
            return result
            
        except Exception as e:
            self.logger.error(f"정규식 키워드 추출 실패: {e}")
            return None



    async def _extract_image_semantics_with_layout_patterns_batch(self, images: List[Dict]) -> List[Dict]:
        """AI Search 레이아웃 패턴을 참조한 이미지 의미 추출 (배치 처리)"""
        
        # 세마포어로 동시 처리 제한
        semaphore = asyncio.Semaphore(5)  # 최대 5개 동시 처리
        
        async def process_single_image(i: int, image: Dict) -> Dict:
            async with semaphore:
                try:
                    layout_patterns = await self._search_image_layout_patterns(image)
                    return await self._analyze_image_with_layout_patterns(image, i, layout_patterns)
                except Exception as e:
                    self.logger.error(f"이미지 {i} 배치 처리 실패: {e}")
                    return self._get_clean_image_fallback(i, image.get("image_name", f"image_{i}"),
                                                       image.get("location", ""), image.get("image_url", ""))
        
        # 병렬 처리
        tasks = [process_single_image(i, image) for i, image in enumerate(images)]
        image_semantics = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 예외 처리된 결과 필터링
        valid_results = []
        for i, result in enumerate(image_semantics):
            if isinstance(result, Exception):
                self.logger.error(f"이미지 {i} 처리 예외: {result}")
                valid_results.append(self._get_clean_image_fallback(i, f"image_{i}", "", ""))
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def _extract_image_semantics_with_layout_patterns(self, images: List[Dict]) -> List[Dict]:
        """AI Search 레이아웃 패턴을 참조한 이미지 의미 추출"""
        
        image_semantics = []
        
        for i, image in enumerate(images):
            # 이미지별 레이아웃 패턴 검색
            layout_patterns = await self._search_image_layout_patterns(image)
            
            semantic_info = await self._analyze_image_with_layout_patterns(image, i, layout_patterns)
            image_semantics.append(semantic_info)
        
        return image_semantics
    
    async def _search_image_layout_patterns(self, image: Dict) -> List[Dict]:
        """이미지 배치를 위한 AI Search 레이아웃 패턴 검색"""
        
        try:
            image_location = image.get("location", "")
            image_name = image.get("image_name", "")
            
            # 이미지 특성 기반 쿼리 생성
            search_query = f"이미지 배치 레이아웃 {image_location} {image_name}"
            clean_query = self.isolation_manager.clean_query_from_azure_keywords(search_query)
            
            # 레이아웃 패턴 검색
            layout_patterns = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.vector_manager.search_similar_layouts(
                    clean_query, "magazine-vector-index", top_k=5
                )
            )
            
            # 격리된 패턴만 반환
            isolated_patterns = self.isolation_manager.filter_contaminated_data(
                layout_patterns, "image_layout_patterns"
            )
            
            return isolated_patterns
            
        except Exception as e:
            self.logger.error(f"이미지 레이아웃 패턴 검색 실패: {e}")
            return []
    
    async def _analyze_image_with_layout_patterns(self, image: Dict, index: int, patterns: List[Dict]) -> Dict:
        """레이아웃 패턴을 참조한 이미지 분석"""
        
        image_name = image.get("image_name", f"image_{index}")
        location = image.get("location", "")
        image_url = image.get("image_url", "")
        
        # 패턴 기반 레이아웃 정보 생성
        layout_context = ""
        if patterns:
            layout_info = []
            for pattern in patterns[:3]:
                layout_info.append({
                    "이미지_크기": pattern.get("image_size", "중간"),
                    "배치_위치": pattern.get("placement", "상단"),
                    "텍스트_간격": pattern.get("text_spacing", "적당함"),
                    "이미지_개수": pattern.get("image_count", 1)
                })
            layout_context = f"레이아웃 참조: {json.dumps(layout_info, ensure_ascii=False)}"
        
        analysis_prompt = f"""
다음 이미지 정보를 바탕으로 의미적 요소를 분석하세요:

이미지명: {image_name}
위치 정보: {location}

{layout_context}

분석 항목:
1. 지리적 특성 (도시, 자연, 건물 등)
2. 시각적 특징 (색상, 구도, 분위기 등)
3. 문화적 맥락
4. 감정적 임팩트
5. 시간대/계절 추정
6. 적합한 이미지 크기 (작은/중간/큰)
7. 권장 배치 위치 (상단/중간/하단)
8. 텍스트와의 적정 간격

반드시 JSON 형식으로 출력하세요.
"""
        
        try:
            response = await self.llm.ainvoke(analysis_prompt)
            
            # ✅ 동일한 강화된 JSON 처리 로직 적용
            if not response or not response.strip():
                self.logger.warning(f"이미지 {index} 분석에서 빈 응답 수신")
                return self._get_clean_image_fallback(index, image_name, location, image_url)
            
            cleaned_response = self._extract_json_from_response(response.strip())
            validated_json = self._validate_and_fix_json(cleaned_response)
            
            try:
                analysis_result = json.loads(validated_json)
            except json.JSONDecodeError as json_error:
                self.logger.error(f"이미지 {index} JSON 파싱 최종 실패: {json_error}")
                # 이미지용 정규식 키워드 추출도 구현 가능
                return self._get_clean_image_fallback(index, image_name, location, image_url)
            
            return {
                "image_index": index,
                "image_name": image_name,
                "location": location,
                "image_url": image_url,
                "semantic_analysis": analysis_result,
                "confidence_score": 0.8,
                "layout_patterns": len(patterns),
                "isolation_metadata": {
                    "patterns_referenced": len(patterns),
                    "contamination_detected": False
                }
            }
            
        except Exception as e:
            self.logger.error(f"이미지 {index} 분석 실패: {e}")
            return self._get_clean_image_fallback(index, image_name, location, image_url)
    
    async def _perform_semantic_matching_with_vectors(self, text_semantics: List[Dict],
                                                    image_semantics: List[Dict]) -> List[Dict]:
        """벡터 데이터 기반 의미적 매칭"""
        
        mappings = []
        
        for text_section in text_semantics:
            section_mappings = []
            
            for image in image_semantics:
                # AI Search 패턴을 고려한 유사도 계산
                similarity_score = await self._calculate_semantic_similarity_with_patterns(
                    text_section, image
                )
                
                section_mappings.append({
                    "image_index": image["image_index"],
                    "image_name": image["image_name"],
                    "similarity_score": similarity_score,
                    "matching_factors": self._identify_matching_factors_with_patterns(text_section, image),
                    "layout_recommendation": self._get_layout_recommendation(text_section, image)
                })
            
            # 유사도 순으로 정렬
            section_mappings.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            mappings.append({
                "text_section_index": text_section["section_index"],
                "text_title": text_section["title"],
                "image_matches": section_mappings[:5]
            })
        
        return mappings
    
    def _extract_json_from_response(self, response: str) -> str:
        """LLM 응답에서 JSON 부분만 추출 (강화된 버전)"""
        
        if not response or not response.strip():
            return "{}"  # 빈 응답 시 기본 JSON 반환
        
        response = response.strip()
        
        # 1. 마크다운 코드 블록 제거 (다양한 패턴 지원)
        patterns = [
            r'``````',  # ``````
            r'``````',      # ``````
            r'`(.*?)`',                # `...`
        ]
        
        for pattern in patterns:
            import re
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if extracted and (extracted.startswith('{') or extracted.startswith('[')):
                    return extracted
        
        # 2. HTML/XML 태그 제거
        if response.startswith('<'):
            # HTML/XML 응답인 경우 JSON 부분 찾기
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json_match.group(0)
            else:
                return "{}"  # JSON을 찾을 수 없으면 빈 객체
        
        # 3. 첫 번째와 마지막 중괄호 사이의 내용 추출
        first_brace = response.find('{')
        last_brace = response.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            potential_json = response[first_brace:last_brace + 1]
            return potential_json
        
        # 4. 배열 형태 JSON 처리
        first_bracket = response.find('[')
        last_bracket = response.rfind(']')
        
        if first_bracket != -1 and last_bracket != -1 and first_bracket < last_bracket:
            potential_json = response[first_bracket:last_bracket + 1]
            return potential_json
        
        # 5. 모든 방법이 실패하면 원본 반환
        return response.strip()

    def _validate_and_fix_json(self, json_str: str) -> str:
        """JSON 문자열 검증 및 수정"""
        
        try:
            # 1. 기본 JSON 파싱 시도
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            pass
        
        # 2. 일반적인 JSON 오류 수정 시도
        fixed_json = json_str
        
        # 따옴표 문제 수정
        fixed_json = fixed_json.replace("'", '"')  # 단일 따옴표를 이중 따옴표로
        fixed_json = fixed_json.replace('True', 'true')  # Python bool을 JSON bool로
        fixed_json = fixed_json.replace('False', 'false')
        fixed_json = fixed_json.replace('None', 'null')
        
        # 마지막 쉼표 제거
        import re
        fixed_json = re.sub(r',\s*}', '}', fixed_json)
        fixed_json = re.sub(r',\s*]', ']', fixed_json)
        
        try:
            json.loads(fixed_json)
            return fixed_json
        except json.JSONDecodeError:
            pass
        
        # 3. 더 공격적인 수정 시도
        try:
            # 키에 따옴표 추가
            fixed_json = re.sub(r'(\w+):', r'"\1":', fixed_json)
            json.loads(fixed_json)
            return fixed_json
        except:
            pass
        
        # 4. 모든 수정 시도 실패 시 기본 JSON 반환
        return '{"error": "JSON 파싱 실패", "original": "' + json_str.replace('"', '\\"')[:100] + '"}'




    async def _calculate_semantic_similarity_with_patterns(self, text_section: Dict, image: Dict) -> float:
        """AI Search 패턴을 고려한 의미적 유사도 계산"""
        
        text_analysis = text_section.get("semantic_analysis", {})
        image_analysis = image.get("semantic_analysis", {})
        
        similarity_factors = []
        
        # 기본 의미적 매칭
        text_keywords = text_analysis.get("시각적_키워드", [])
        image_features = image_analysis.get("지리적_특성", [])
        keyword_match = len(set(text_keywords) & set(image_features)) / max(len(text_keywords), 1)
        similarity_factors.append(keyword_match * 0.3)
        
        # 감정적 톤 매칭
        text_tone = text_analysis.get("감정적_톤", "")
        image_impact = image_analysis.get("감정적_임팩트", "")
        tone_match = 1.0 if text_tone == image_impact else 0.5
        similarity_factors.append(tone_match * 0.2)
        
        # AI Search 패턴 기반 레이아웃 적합성
        layout_compatibility = self._calculate_layout_compatibility(text_section, image)
        similarity_factors.append(layout_compatibility * 0.3)
        
        # 문화적/지리적 연관성
        cultural_match = self._calculate_cultural_relevance(text_analysis, image_analysis)
        similarity_factors.append(cultural_match * 0.2)
        
        # 전체 유사도 계산
        total_similarity = sum(similarity_factors)
        
        # 신뢰도 가중치 적용
        confidence_weight = (text_section.get("confidence_score", 0.5) + 
                           image.get("confidence_score", 0.5)) / 2
        
        return min(total_similarity * confidence_weight, 1.0)
    
    def _calculate_layout_compatibility(self, text_section: Dict, image: Dict) -> float:
        """레이아웃 호환성 계산"""
        
        text_patterns = text_section.get("ai_search_patterns", 0)
        image_patterns = image.get("layout_patterns", 0)
        
        # 패턴 데이터가 있으면 높은 점수
        if text_patterns > 0 and image_patterns > 0:
            return 0.9
        elif text_patterns > 0 or image_patterns > 0:
            return 0.7
        else:
            return 0.5
    
    def _calculate_cultural_relevance(self, text_analysis: Dict, image_analysis: Dict) -> float:
        """문화적 연관성 계산"""
        
        text_cultural = text_analysis.get("문화적_요소", [])
        image_cultural = image_analysis.get("문화적_맥락", [])
        
        if not text_cultural and not image_cultural:
            return 0.5
        
        common_cultural = set(text_cultural) & set(image_cultural)
        total_cultural = set(text_cultural) | set(image_cultural)
        
        if not total_cultural:
            return 0.5
        
        return len(common_cultural) / len(total_cultural)
    
    def _identify_matching_factors_with_patterns(self, text_section: Dict, image: Dict) -> List[str]:
        """AI Search 패턴을 고려한 매칭 요인 식별"""
        
        factors = []
        
        text_analysis = text_section.get("semantic_analysis", {})
        image_analysis = image.get("semantic_analysis", {})
        
        # 기본 매칭 요인
        text_keywords = set(text_analysis.get("시각적_키워드", []))
        image_features = set(image_analysis.get("지리적_특성", []))
        
        common_elements = text_keywords & image_features
        if common_elements:
            factors.append(f"공통_요소: {', '.join(common_elements)}")
        
        # AI Search 패턴 기반 요인
        if text_section.get("ai_search_patterns", 0) > 0:
            factors.append("텍스트_패턴_참조")
        
        if image.get("layout_patterns", 0) > 0:
            factors.append("레이아웃_패턴_참조")
        
        # 레이아웃 호환성
        layout_score = self._calculate_layout_compatibility(text_section, image)
        if layout_score > 0.8:
            factors.append("레이아웃_고도_호환")
        elif layout_score > 0.6:
            factors.append("레이아웃_적당_호환")
        
        return factors
    
    def _get_layout_recommendation(self, text_section: Dict, image: Dict) -> Dict:
        """레이아웃 추천 생성"""
        
        image_analysis = image.get("semantic_analysis", {})
        
        return {
            "권장_이미지_크기": image_analysis.get("적합한_이미지_크기", "중간"),
            "권장_배치_위치": image_analysis.get("권장_배치_위치", "상단"),
            "텍스트_간격": image_analysis.get("텍스트와의_적정_간격", "적당함"),
            "패턴_기반": text_section.get("ai_search_patterns", 0) > 0 and image.get("layout_patterns", 0) > 0
        }
    
    async def _generate_optimal_combinations_with_ai_search(self, semantic_mappings: List[Dict]) -> List[Dict]:
        """AI Search 데이터를 활용한 최적 조합 생성 (다양성 최적화 인식)"""
        
        optimal_combinations = []
        used_images = set()
        
        for mapping in semantic_mappings:
            section_index = mapping["text_section_index"]
            section_title = mapping["text_title"]
            
            # ✅ 기존 AI Search 패턴 기반 이미지 선택 로직 유지
            best_images = []
            semantic_matches = []  # ✅ 의미적 매칭 정보도 별도 보관
            
            for image_match in mapping["image_matches"]:
                image_index = image_match["image_index"]
                
                # ✅ 중복 방지 (ImageDiversityManager와 협력)
                if image_index not in used_images:
                    semantic_matches.append(image_match)  # 의미적 매칭 정보 보관
                    
                    # ✅ 기존 AI Search 패턴 우선순위 로직 유지
                    layout_rec = image_match.get("layout_recommendation", {})
                    if layout_rec.get("패턴_기반", False):
                        best_images.insert(0, image_match)  # 앞에 추가
                    else:
                        best_images.append(image_match)
                    
                    used_images.add(image_index)
                    
                    # ✅ 기존 제한 로직 유지
                    if len(best_images) >= 3:
                        break
            
            # ✅ 기존 구조 유지 + 다양성 정보 추가
            optimal_combinations.append({
                "section_index": section_index,
                "section_title": section_title,
                "assigned_images": best_images,  # ✅ 기존 키 유지 (호환성)
                "semantic_matches": semantic_matches,  # ✅ 추가 정보
                "total_similarity_score": sum(img["similarity_score"] for img in best_images),
                "semantic_score": sum(match.get("similarity_score", 0) for match in semantic_matches),
                "ai_search_enhanced": any(img.get("layout_recommendation", {}).get("패턴_기반", False) for img in best_images),
                "diversity_aware": True,  # ✅ 다양성 인식 표시
                "optimization_notes": f"{len(best_images)}개 이미지 할당됨 (AI Search 패턴 + 다양성 인식)"
            })
        
        self.logger.info(f"의미적 분석 완료: {len(optimal_combinations)}개 조합, "
                        f"{len(used_images)}개 고유 이미지 (AI Search 패턴 + 다양성 최적화)")
        
        return optimal_combinations
    
    # 기존 헬퍼 메서드들 유지
    def _get_clean_analysis_fallback(self) -> Dict:
        """오염되지 않은 기본 분석 결과"""
        return {
            "주요_주제": ["여행"],
            "감정적_톤": "중성적",
            "시각적_키워드": ["풍경"],
            "계절_시간": "알 수 없음",
            "문화적_요소": [],
            "글의_형태": "서술형",
            "문장_길이_특성": "중간"
        }
    
    def _get_clean_section_fallback(self, index: int, title: str, content: str) -> Dict:
        """오염되지 않은 기본 섹션 결과"""
        return {
            "section_index": index,
            "title": title,
            "content_preview": content[:200],
            "semantic_analysis": self._get_clean_analysis_fallback(),
            "confidence_score": 0.3,
            "ai_search_patterns": 0,
            "isolation_metadata": {
                "patterns_referenced": 0,
                "contamination_detected": True,
                "fallback_used": True
            }
        }
    
    def _get_clean_image_fallback(self, index: int, image_name: str, location: str, image_url: str) -> Dict:
        """오염되지 않은 기본 이미지 결과"""
        return {
            "image_index": index,
            "image_name": image_name,
            "location": location,
            "image_url": image_url,
            "semantic_analysis": {
                "지리적_특성": ["도시"],
                "시각적_특징": ["일반적"],
                "문화적_맥락": [],
                "감정적_임팩트": "중성적",
                "시간대_계절": "알 수 없음",
                "적합한_이미지_크기": "중간",
                "권장_배치_위치": "상단",
                "텍스트와의_적정_간격": "적당함"
            },
            "confidence_score": 0.3,
            "layout_patterns": 0,
            "isolation_metadata": {
                "patterns_referenced": 0,
                "contamination_detected": True
            }
        }
    
    def _generate_clean_fallback_result(self, content: Dict, images: List[Dict]) -> Dict:
        """완전히 정화된 폴백 결과"""
        return {
            "text_semantics": [],
            "image_semantics": [],
            "semantic_mappings": [],
            "optimal_combinations": [],
            "analysis_metadata": {
                "total_text_sections": 0,
                "total_images": 0,
                "mapping_confidence": 0.0,
                "ai_search_enhanced": False,
                "isolation_applied": True,
                "contamination_detected": True,
                "fallback_used": True
            }
        }
    
    def _calculate_overall_confidence(self, semantic_mappings: List[Dict]) -> float:
        """전체 매핑 신뢰도 계산"""
        
        if not semantic_mappings:
            return 0.0
        
        total_confidence = 0.0
        total_matches = 0
        
        for mapping in semantic_mappings:
            for image_match in mapping["image_matches"]:
                total_confidence += image_match["similarity_score"]
                total_matches += 1
        
        return total_confidence / max(total_matches, 1)
