import asyncio
import numpy as np
import time
from typing import List, Dict, Set, Optional
from PIL import Image
import requests
from io import BytesIO
import imagehash
import cv2
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from utils.log.hybridlogging import HybridLogger
from utils.isolation.session_isolation import SessionAwareMixin
from utils.log.logging_manager import LoggingManager
from utils.data.pdf_vector_manager import PDFVectorManager
from utils.isolation.ai_search_isolation import AISearchIsolationManager
import onnxruntime as ort
import os

class ImageDiversityManager(SessionAwareMixin):
    """이미지 다양성 관리 및 중복 방지 전문 에이전트 - 벡터 통합 버전"""
    
    def __init__(self, vector_manager: PDFVectorManager, logger: 'HybridLogger', 
                 similarity_threshold: int = 40, diversity_weight: float = 0.3):
        super().__init__()
        self.logger = logger
        self.logging_manager = LoggingManager(self.logger)
        
        # ✅ 벡터 매니저 통합
        self.vector_manager = vector_manager
        self.isolation_manager = AISearchIsolationManager()
        
        self.similarity_threshold = similarity_threshold
        self.diversity_weight = diversity_weight
        self.processed_hashes: Set[str] = set()
        self.image_clusters: Dict[str, List[Dict]] = {}
        self.image_embeddings_cache: Dict[str, np.ndarray] = {}
        
        self.__init_session_awareness__()        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # ✅ CLIP 모델 중복 제거 - 외부 ONNX 세션 사용
        self.clip_available = False
        self.onnx_session = None
        self.clip_preprocess = None
        self.device = "cpu"
        
        # ✅ 외부에서 CLIP 모델을 주입받도록 변경
        self._initialize_external_clip_model()
        
        # 품질 평가 메트릭
        self.quality_metrics = {
            "sharpness": self._calculate_sharpness,
            "contrast": self._calculate_contrast,
            "brightness": self._calculate_brightness,
            "composition": self._calculate_composition_score
        }
        
        self.logger.info("ImageDiversityManager 초기화 완료 (벡터 통합 버전)")

    def _initialize_external_clip_model(self):
        """외부 CLIP 모델 초기화 (중복 방지)"""
        onnx_model_path = "models/clip_onnx/clip_visual.quant.onnx"
        
        if os.path.exists(onnx_model_path):
            try:
                self.onnx_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
                
                # 전처리기만 open_clip에서 가져옴 (중복 방지)
                import open_clip
                _, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32', pretrained='laion2b_s34b_b79k', device=self.device
                )
                self.clip_available = True
                self.logger.info("✅ ImageDiversityManager: 외부 ONNX CLIP 모델 연결 성공")
            except Exception as e:
                self.logger.warning(f"외부 ONNX 모델 연결 실패: {e}")
        else:
            self.logger.warning(f"ONNX 모델 파일을 찾을 수 없습니다: {onnx_model_path}")

    def set_external_clip_session(self, onnx_session, clip_preprocess):
        """외부에서 CLIP 세션 주입 (중복 방지)"""
        self.onnx_session = onnx_session
        self.clip_preprocess = clip_preprocess
        self.clip_available = True if onnx_session else False
        self.logger.info("✅ 외부 CLIP 세션 주입 완료")

    async def process_data(self, input_data):
        result = await self._do_work(input_data)
        
        await self.logging_manager.log_agent_response(
            agent_name=self.__class__.__name__,
            agent_role="벡터 통합 이미지 다양성 관리 에이전트",
            task_description="벡터 검색 기반 이미지 다양성 최적화",
            response_data=result, 
            metadata={"vector_integrated": True, "clip_shared": True}
        )
        
        return result

    async def optimize_image_distribution(self, images: List[Dict], sections: List[Dict], 
                                        unified_patterns: Optional[Dict] = None) -> Dict:
        """✅ 벡터 통합 기반 이미지 분배 최적화"""
        self.logger.info(f"벡터 통합 이미지 다양성 최적화 시작: {len(images)}개 이미지, {len(sections)}개 섹션")
        
        try:
            # 1. 중복 이미지 제거
            unique_images = await self._remove_duplicate_images_async(images)
            
            # 2. 이미지 품질 평가
            quality_enhanced_images = await self._enhance_image_quality_scores(unique_images)
            
            # ✅ 3. 벡터 검색 기반 이미지 의미 패턴 수집
            if unified_patterns:
                semantic_patterns = await self._collect_image_semantic_patterns(
                    quality_enhanced_images, unified_patterns
                )
            else:
                semantic_patterns = {}
            
            # 4. CLIP 기반 이미지 클러스터링 (가능한 경우)
            if self.clip_available and len(quality_enhanced_images) > 5:
                clustered_images = await self._cluster_images_with_clip_and_vectors(
                    quality_enhanced_images, semantic_patterns
                )
            else:
                clustered_images = {"default_cluster": quality_enhanced_images}
            
            # 5. 벡터 패턴 기반 대표 이미지 선택
            representative_images = self._select_representative_images_with_vectors(
                clustered_images, semantic_patterns
            )
            
            # 6. 벡터 다양성을 고려한 섹션별 이미지 할당
            allocation_plan = await self._allocate_images_with_vector_diversity(
                representative_images, sections, semantic_patterns
            )
            
            # 7. 결과 로깅 및 저장
            await self._log_diversity_optimization_results(allocation_plan, images, sections)
            
            self.logger.info(f"벡터 통합 이미지 최적화 완료: {len(allocation_plan)}개 섹션에 할당")
            
            return {
                "allocation_plan": allocation_plan,
                "embeddings_cache": self.image_embeddings_cache,
                "semantic_patterns": semantic_patterns,
                "vector_enhanced": True
            }
            
        except Exception as e:
            self.logger.error(f"벡터 통합 이미지 최적화 실패: {e}")
            # 폴백: 기본 순차 분배
            fallback_plan = self._fallback_sequential_distribution(images, sections)
            return {
                "allocation_plan": fallback_plan,
                "embeddings_cache": self.image_embeddings_cache,
                "vector_enhanced": False
            }

    async def _collect_image_semantic_patterns(self, images: List[Dict], 
                                             unified_patterns: Dict) -> Dict:
        """✅ 벡터 검색 기반 이미지 의미 패턴 수집"""
        try:
            image_patterns = {}
            
            for i, image_data in enumerate(images):
                # 이미지 설명 기반 쿼리 생성
                description = image_data.get("description", "")
                location = image_data.get("city", "") + " " + image_data.get("country", "")
                query = f"image {description} {location}".strip()
                
                if not query or query == "image":
                    continue
                
                # 3개 벡터 인덱스에서 검색
                patterns = await self._search_cross_index_patterns(query)
                
                image_patterns[f"image_{i}"] = {
                    "query": query,
                    "patterns": patterns,
                    "semantic_score": self._calculate_pattern_relevance(patterns)
                }
            
            return image_patterns
            
        except Exception as e:
            self.logger.error(f"이미지 의미 패턴 수집 실패: {e}")
            return {}

    async def _search_cross_index_patterns(self, query: str) -> Dict:
        """✅ 3개 벡터 인덱스 교차 검색"""
        try:
            clean_query = self.isolation_manager.clean_query_from_azure_keywords(query)
            
            # 병렬로 3개 인덱스 검색
            tasks = [
                self._search_magazine_patterns(clean_query),
                self._search_jsx_patterns(clean_query),
                self._search_semantic_patterns(clean_query)
            ]
            
            magazine_results, jsx_results, semantic_results = await asyncio.gather(*tasks)
            
            return {
                "magazine_patterns": magazine_results,
                "jsx_patterns": jsx_results,
                "semantic_patterns": semantic_results,
                "cross_index_score": self._calculate_cross_index_alignment(
                    magazine_results, jsx_results, semantic_results
                )
            }
            
        except Exception as e:
            self.logger.error(f"교차 인덱스 검색 실패: {e}")
            return {"magazine_patterns": [], "jsx_patterns": [], "semantic_patterns": []}

    async def _search_magazine_patterns(self, query: str) -> List[Dict]:
        """매거진 벡터 인덱스 검색"""
        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.vector_manager.search_similar_layouts(
                    query, "magazine-vector-index", top_k=3
                )
            )
            return self.isolation_manager.filter_contaminated_data(
                results, f"magazine_patterns_{hash(query)}"
            )
        except Exception as e:
            self.logger.error(f"매거진 패턴 검색 실패: {e}")
            return []

    async def _search_jsx_patterns(self, query: str) -> List[Dict]:
        """JSX 컴포넌트 벡터 인덱스 검색"""
        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.vector_manager.search_similar_layouts(
                    query, "jsx-component-vector-index", top_k=3
                )
            )
            return self.isolation_manager.filter_contaminated_data(
                results, f"jsx_patterns_{hash(query)}"
            )
        except Exception as e:
            self.logger.error(f"JSX 패턴 검색 실패: {e}")
            return []

    async def _search_semantic_patterns(self, query: str) -> List[Dict]:
        """텍스트 의미 벡터 인덱스 검색"""
        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.vector_manager.search_similar_layouts(
                    query, "text-semantic-patterns-index", top_k=3
                )
            )
            return self.isolation_manager.filter_contaminated_data(
                results, f"semantic_patterns_{hash(query)}"
            )
        except Exception as e:
            self.logger.error(f"의미 패턴 검색 실패: {e}")
            return []

    def _calculate_cross_index_alignment(self, magazine_results: List[Dict], 
                                       jsx_results: List[Dict], 
                                       semantic_results: List[Dict]) -> float:
        """교차 인덱스 정렬도 계산"""
        total_results = len(magazine_results) + len(jsx_results) + len(semantic_results)
        if total_results == 0:
            return 0.0
        
        # 간단한 정렬도 계산 (실제로는 더 복잡한 로직 필요)
        alignment_score = min(total_results / 9.0, 1.0)  # 최대 9개 결과 기준
        return alignment_score

    def _calculate_pattern_relevance(self, patterns: Dict) -> float:
        """패턴 관련성 점수 계산"""
        cross_index_score = patterns.get("cross_index_score", 0.0)
        pattern_count = (
            len(patterns.get("magazine_patterns", [])) +
            len(patterns.get("jsx_patterns", [])) +
            len(patterns.get("semantic_patterns", []))
        )
        
        relevance_score = (cross_index_score * 0.7) + (min(pattern_count / 6.0, 1.0) * 0.3)
        return relevance_score

    async def _cluster_images_with_clip_and_vectors(self, images: List[Dict], 
                                                   semantic_patterns: Dict) -> Dict[str, List[Dict]]:
        """✅ CLIP + 벡터 패턴 기반 이미지 클러스터링"""
        if not self.clip_available or len(images) < 3:
            return {"default_cluster": images}
        
        try:
            self.logger.info(f"CLIP + 벡터 기반 클러스터링 시작: {len(images)}개 이미지")
            
            # CLIP 임베딩 생성
            clip_embeddings = await self._generate_clip_embeddings(images)
            
            if clip_embeddings is None or len(clip_embeddings) == 0:
                return {"default_cluster": images}
            
            # 벡터 패턴 점수를 CLIP 임베딩에 통합
            enhanced_embeddings = self._enhance_embeddings_with_vector_patterns(
                clip_embeddings, images, semantic_patterns
            )
            
            # DBSCAN 클러스터링 수행
            clustering = DBSCAN(eps=0.7, min_samples=1, metric='cosine', algorithm='brute')
            cluster_labels = clustering.fit_predict(enhanced_embeddings)
            
            # 클러스터별 이미지 그룹화
            clusters = {}
            for i, label in enumerate(cluster_labels):
                cluster_key = f"cluster_{label}" if label != -1 else "outliers"
                
                if cluster_key not in clusters:
                    clusters[cluster_key] = []
                
                clusters[cluster_key].append(images[i])
            
            self.logger.info(f"벡터 강화 클러스터링 완료: {len(clusters)}개 클러스터")
            return clusters
            
        except Exception as e:
            self.logger.error(f"벡터 강화 클러스터링 실패: {e}")
            return {"default_cluster": images}

    def _enhance_embeddings_with_vector_patterns(self, clip_embeddings: np.ndarray, 
                                                images: List[Dict], 
                                                semantic_patterns: Dict) -> np.ndarray:
        """CLIP 임베딩에 벡터 패턴 정보 통합"""
        enhanced_embeddings = clip_embeddings.copy()
        
        for i, image_data in enumerate(images):
            pattern_key = f"image_{i}"
            if pattern_key in semantic_patterns:
                pattern_score = semantic_patterns[pattern_key].get("semantic_score", 0.0)
                
                # 패턴 점수를 임베딩에 가중치로 적용
                weight_factor = 1.0 + (pattern_score * 0.1)  # 최대 10% 가중치
                enhanced_embeddings[i] = enhanced_embeddings[i] * weight_factor
                
                # 정규화
                norm = np.linalg.norm(enhanced_embeddings[i])
                if norm > 0:
                    enhanced_embeddings[i] = enhanced_embeddings[i] / norm
        
        return enhanced_embeddings

    def _select_representative_images_with_vectors(self, clusters: Dict[str, List[Dict]], 
                                                 semantic_patterns: Dict) -> List[Dict]:
        """✅ 벡터 패턴 기반 대표 이미지 선택"""
        representative_images = []
        temp_hashes = set()
        
        for cluster_name, cluster_images in clusters.items():
            # 클러스터 내에서 벡터 패턴 점수가 높은 이미지 우선 선택
            scored_images = []
            
            for img in cluster_images:
                img_hash = img.get("perceptual_hash")
                quality = img.get("overall_quality", 0.5)
                
                # 벡터 패턴 점수 추가
                image_index = cluster_images.index(img)
                pattern_key = f"image_{image_index}"
                vector_score = semantic_patterns.get(pattern_key, {}).get("semantic_score", 0.0)
                
                # 종합 점수 계산
                combined_score = (quality * 0.6) + (vector_score * 0.4)
                
                scored_images.append((img, combined_score, img_hash))
            
            # 점수순으로 정렬
            scored_images.sort(key=lambda x: x[1], reverse=True)
            
            # 상위 이미지들 선택 (중복 제거)
            for img, score, img_hash in scored_images:
                if score >= 0.3 and img_hash and img_hash not in temp_hashes:
                    representative_images.append(img)
                    temp_hashes.add(img_hash)
                elif not img_hash:  # 해시가 없으면 포함
                    representative_images.append(img)
        
        self.logger.info(f"벡터 기반 대표 이미지 선택 완료: {len(representative_images)}개")
        return representative_images

    async def _allocate_images_with_vector_diversity(self, images: List[Dict], 
                                                   sections: List[Dict], 
                                                   semantic_patterns: Dict) -> Dict:
        """✅ 벡터 다양성 기반 섹션별 이미지 할당"""
        total_images = len(images)
        total_sections = len(sections)
        
        if total_images == 0 or total_sections == 0:
            return {f"section_{i}": {
                "images": [], "count": 0, 
                "section_title": sections[i].get("title", f"섹션 {i+1}") if i < len(sections) else f"섹션 {i+1}",
                "diversity_score": 0.0, "avg_quality": 0.0, "vector_score": 0.0
            } for i in range(max(total_sections, 1))}

        # CLIP 임베딩 생성 (벡터 패턴 강화)
        if self.clip_available:
            image_embeddings = await self._generate_clip_embeddings(images)
            if image_embeddings is not None:
                image_embeddings = self._enhance_embeddings_with_vector_patterns(
                    image_embeddings, images, semantic_patterns
                )
        else:
            image_embeddings = None

        allocation_plan = {}
        assigned_image_indices = set()
        
        # 섹션별 필요 이미지 수 계산
        images_per_section = max(1, total_images // total_sections) if total_sections > 0 else 1
        
        for i, section in enumerate(sections):
            # 할당량 계산
            allocation_count = images_per_section
            if i < total_images % total_sections:
                allocation_count += 1
            
            # 남은 이미지 중에서 할당
            available_indices = [idx for idx in range(total_images) if idx not in assigned_image_indices]
            
            # 벡터 패턴 + 품질 + 다양성 기반 선택
            if available_indices:
                selected_indices = self._select_images_with_vector_diversity(
                    images, available_indices, allocation_count, 
                    semantic_patterns, image_embeddings
                )
            else:
                selected_indices = []

            selected_images = [images[idx] for idx in selected_indices]
            for idx in selected_indices:
                assigned_image_indices.add(idx)

            # 섹션 다양성 점수 계산 (벡터 강화)
            section_diversity_score = self._calculate_vector_enhanced_diversity(
                selected_images, semantic_patterns, image_embeddings, selected_indices
            )
            
            # 벡터 패턴 점수 계산
            vector_score = self._calculate_section_vector_score(selected_images, semantic_patterns)

            allocation_plan[f"section_{i}"] = {
                "images": selected_images,
                "count": len(selected_images),
                "section_title": section.get("title", f"섹션 {i+1}"),
                "diversity_score": section_diversity_score,
                "avg_quality": float(np.mean([img.get("overall_quality", 0.5) for img in selected_images])) if selected_images else 0.0,
                "vector_score": vector_score
            }

        return allocation_plan

    def _select_images_with_vector_diversity(self, images: List[Dict], available_indices: List[int],
                                           allocation_count: int, semantic_patterns: Dict,
                                           image_embeddings: Optional[np.ndarray]) -> List[int]:
        """벡터 다양성 기반 이미지 선택"""
        if len(available_indices) <= allocation_count:
            return available_indices
        
        # 종합 점수 계산 (품질 + 벡터 패턴)
        scored_indices = []
        for idx in available_indices:
            quality_score = images[idx].get("overall_quality", 0.5)
            
            # 벡터 패턴 점수
            pattern_key = f"image_{idx}"
            vector_score = semantic_patterns.get(pattern_key, {}).get("semantic_score", 0.0)
            
            # 종합 점수
            combined_score = (quality_score * 0.6) + (vector_score * 0.4)
            scored_indices.append((idx, combined_score))
        
        # 점수순 정렬 후 상위 선택
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, _ in scored_indices[:allocation_count]]
        
        return selected_indices

    def _calculate_vector_enhanced_diversity(self, selected_images: List[Dict], 
                                           semantic_patterns: Dict,
                                           image_embeddings: Optional[np.ndarray],
                                           selected_indices: List[int]) -> float:
        """벡터 강화 다양성 점수 계산"""
        if len(selected_images) <= 1:
            return 1.0
        
        # CLIP 기반 다양성
        clip_diversity = 0.5
        if image_embeddings is not None and selected_indices:
            selected_embeddings = image_embeddings[selected_indices]
            clip_diversity = self._calculate_section_diversity(selected_embeddings)
        
        # 벡터 패턴 기반 다양성
        vector_diversity = self._calculate_vector_pattern_diversity(selected_images, semantic_patterns)
        
        # 종합 다양성 점수
        enhanced_diversity = (clip_diversity * 0.7) + (vector_diversity * 0.3)
        return enhanced_diversity

    def _calculate_vector_pattern_diversity(self, selected_images: List[Dict], 
                                          semantic_patterns: Dict) -> float:
        """벡터 패턴 기반 다양성 계산"""
        if len(selected_images) <= 1:
            return 1.0
        
        pattern_scores = []
        for i, img in enumerate(selected_images):
            pattern_key = f"image_{i}"
            pattern_data = semantic_patterns.get(pattern_key, {})
            
            # 각 인덱스별 패턴 수
            magazine_count = len(pattern_data.get("patterns", {}).get("magazine_patterns", []))
            jsx_count = len(pattern_data.get("patterns", {}).get("jsx_patterns", []))
            semantic_count = len(pattern_data.get("patterns", {}).get("semantic_patterns", []))
            
            pattern_scores.append([magazine_count, jsx_count, semantic_count])
        
        if not pattern_scores:
            return 0.5
        
        # 패턴 다양성 계산 (표준편차 기반)
        pattern_array = np.array(pattern_scores)
        diversity_score = np.mean(np.std(pattern_array, axis=0))
        
        return min(diversity_score / 3.0, 1.0)  # 정규화

    def _calculate_section_vector_score(self, selected_images: List[Dict], 
                                      semantic_patterns: Dict) -> float:
        """섹션의 벡터 패턴 점수 계산"""
        if not selected_images:
            return 0.0
        
        total_score = 0.0
        for i, img in enumerate(selected_images):
            pattern_key = f"image_{i}"
            pattern_score = semantic_patterns.get(pattern_key, {}).get("semantic_score", 0.0)
            total_score += pattern_score
        
        return total_score / len(selected_images)

    # 기존 메서드들 유지 (중복 제거, 품질 평가, CLIP 임베딩 등)
    async def _remove_duplicate_images_async(self, images: List[Dict]) -> List[Dict]:
        """비동기 중복 이미지 제거 (기존 로직 유지)"""
        unique_images = []
        
        for image_data in images:
            try:
                image_url = image_data.get("image_url", "")
                image_name = image_data.get("image_name", "")
                
                if not image_url:
                    continue
                
                image_hash = await self._calculate_perceptual_hash_async(image_url)
                
                if image_hash and not self._is_duplicate_or_similar(image_hash):
                    self.processed_hashes.add(image_hash)
                    image_data["perceptual_hash"] = image_hash
                    unique_images.append(image_data)
                    self.logger.debug(f"고유 이미지 추가: {image_name}")
                else:
                    self.logger.debug(f"중복 이미지 제거: {image_name}")
                    
            except Exception as e:
                self.logger.error(f"이미지 중복 검사 실패 {image_data.get('image_name', 'Unknown')}: {e}")
                unique_images.append(image_data)
        
        self.logger.info(f"중복 제거 완료: {len(images)} → {len(unique_images)}개")
        return unique_images

    async def _calculate_perceptual_hash_async(self, image_url: str) -> Optional[str]:
        """비동기 Perceptual Hash 계산"""
        try:
            loop = asyncio.get_event_loop()
            image_hash = await loop.run_in_executor(
                None, self._calculate_perceptual_hash_sync, image_url
            )
            return image_hash
        except Exception as e:
            self.logger.error(f"Perceptual hash 계산 실패: {e}")
            return None

    def _calculate_perceptual_hash_sync(self, image_url: str) -> str:
        """동기 Perceptual Hash 계산"""
        try:
            if image_url.startswith(('http://', 'https://')):
                response = self.session.get(image_url, timeout=30)
                response.raise_for_status()
                
                content_type = response.headers.get('content-type', '')
                if not content_type.startswith('image/'):
                    self.logger.warning(f"URL이 이미지가 아님: {content_type}")
                    return ""
                
                image_data = BytesIO(response.content)
                with Image.open(image_data) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    phash = imagehash.phash(img, hash_size=16)
                    return str(phash)
            else:
                with Image.open(image_url) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    phash = imagehash.phash(img, hash_size=16)
                    return str(phash)
                    
        except Exception as e:
            self.logger.error(f"이미지 해시 계산 실패 {image_url}: {e}")
            return ""

    def _is_duplicate_or_similar(self, image_hash: str) -> bool:
        """중복 또는 유사 이미지 검사"""
        try:
            current_hash = imagehash.hex_to_hash(image_hash)
            
            for existing_hash_str in self.processed_hashes:
                existing_hash = imagehash.hex_to_hash(existing_hash_str)
                distance = current_hash - existing_hash
                
                if distance <= self.similarity_threshold:
                    if distance <= 5:
                        return True
                    return False
            
            return False
        except Exception as e:
            self.logger.error(f"유사도 검사 실패: {e}")
            return False

    async def _enhance_image_quality_scores(self, images: List[Dict]) -> List[Dict]:
        """이미지 품질 점수 향상"""
        enhanced_images = []
        
        for image_data in images:
            try:
                image_url = image_data.get("image_url", "")
                
                if image_url:
                    quality_scores = await self._assess_image_quality_async(image_url)
                    image_data["quality_scores"] = quality_scores
                    image_data["overall_quality"] = quality_scores.get("overall", 0.5)
                else:
                    image_data["quality_scores"] = {"overall": 0.3}
                    image_data["overall_quality"] = 0.3
                
                enhanced_images.append(image_data)
                
            except Exception as e:
                self.logger.error(f"품질 점수 계산 실패: {e}")
                image_data["quality_scores"] = {"overall": 0.3}
                image_data["overall_quality"] = 0.3
                enhanced_images.append(image_data)
        
        return enhanced_images

    async def _assess_image_quality_async(self, image_url: str) -> Dict[str, float]:
        """비동기 이미지 품질 평가"""
        try:
            loop = asyncio.get_event_loop()
            quality_scores = await loop.run_in_executor(
                None, self._assess_image_quality_sync, image_url
            )
            return quality_scores
        except Exception as e:
            self.logger.error(f"이미지 품질 평가 실패: {e}")
            return {"overall": 0.3, "error": str(e)}

    def _assess_image_quality_sync(self, image_url: str) -> Dict[str, float]:
        """동기 이미지 품질 평가"""
        return {
            "overall": 0.75,
            "sharpness": 0.7,
            "contrast": 0.7,
            "brightness": 0.7,
            "composition": 0.7,
            "note": "Simplified for performance"
        }

    async def _generate_clip_embeddings(self, images: List[Dict]) -> Optional[np.ndarray]:
        """CLIP 임베딩 생성 (외부 세션 사용)"""
        try:
            if not self.clip_available or not self.onnx_session:
                return None
                
            embeddings = []
            
            for image_data in images:
                image_url = image_data.get("image_url", "")
                
                if image_url in self.image_embeddings_cache:
                    embeddings.append(self.image_embeddings_cache[image_url])
                    continue
                
                try:
                    if image_url.startswith(('http://', 'https://')):
                        response = self.session.get(image_url, timeout=30)
                        response.raise_for_status()
                        
                        image_data_bytes = BytesIO(response.content)
                        pil_image = Image.open(image_data_bytes)
                    else:
                        pil_image = Image.open(image_url)
                    
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    
                    # ✅ 외부 ONNX 세션 사용
                    image_input_tensor = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
                    image_input_np = image_input_tensor.cpu().numpy()

                    onnx_inputs = {self.onnx_session.get_inputs()[0].name: image_input_np}
                    onnx_outputs = self.onnx_session.run(None, onnx_inputs)
                    embedding = onnx_outputs[0].flatten()

                    norm = np.linalg.norm(embedding)
                    normalized_embedding = embedding / norm if norm != 0 else embedding
                    
                    self.image_embeddings_cache[image_url] = normalized_embedding
                    embeddings.append(normalized_embedding)
                        
                except Exception as e:
                    self.logger.error(f"ONNX 기반 임베딩 생성 실패 {image_url}: {e}")
                    embeddings.append(np.zeros(512))
            
            return np.array(embeddings) if embeddings else None
            
        except Exception as e:
            self.logger.error(f"CLIP 임베딩 생성 전체 실패: {e}")
            return None

    def _calculate_section_diversity(self, embeddings: Optional[np.ndarray]) -> float:
        """섹션 내 이미지 다양성 점수 계산"""
        if embeddings is None or len(embeddings) <= 1:
            return 1.0
        
        try:
            cosine_similarity_matrix = np.dot(embeddings, embeddings.T)
            num_embeddings = len(embeddings)
            total_similarity = np.sum(np.triu(cosine_similarity_matrix, k=1))
            pair_count = num_embeddings * (num_embeddings - 1) / 2

            if pair_count == 0:
                return 1.0

            avg_similarity = total_similarity / pair_count
            avg_distance = 1.0 - avg_similarity
            
            return avg_distance
            
        except Exception as e:
            self.logger.error(f"다양성 점수 계산 실패: {e}")
            return 0.5

    def _fallback_sequential_distribution(self, images: List[Dict], 
                                        sections: List[Dict]) -> Dict[str, Dict]:
        """폴백: 순차적 이미지 분배"""
        self.logger.warning("폴백 모드: 순차적 이미지 분배 사용")
        
        total_images = len(images)
        total_sections = len(sections)
        
        if total_images == 0:
            return {f"section_{i}": {
                "images": [], "count": 0, 
                "section_title": f"섹션 {i+1}",
                "diversity_score": 0.5, "avg_quality": 0.5, "vector_score": 0.0
            } for i in range(total_sections)}
        
        base_allocation = max(1, total_images // total_sections)
        remainder = total_images % total_sections
        
        allocation_plan = {}
        current_idx = 0
        
        for i, section in enumerate(sections):
            allocation_count = base_allocation + (1 if i < remainder else 0)
            end_idx = min(current_idx + allocation_count, total_images)
            
            allocated_images = images[current_idx:end_idx]
            
            allocation_plan[f"section_{i}"] = {
                "images": allocated_images,
                "count": len(allocated_images),
                "section_title": section.get("title", f"섹션 {i+1}"),
                "diversity_score": 0.5,
                "avg_quality": 0.5,
                "vector_score": 0.0,
                "fallback_used": True
            }
            
            current_idx = end_idx
        
        return allocation_plan

    async def _log_diversity_optimization_results(self, allocation_plan: Dict, 
                                            original_images: List[Dict], 
                                            sections: List[Dict]) -> None:
        """다양성 최적화 결과 로깅"""
        try:
            total_allocated = sum(data["count"] for data in allocation_plan.values())
            diversity_scores = [data["diversity_score"] for data in allocation_plan.values()]
            quality_scores = [data["avg_quality"] for data in allocation_plan.values()]
            vector_scores = [data.get("vector_score", 0.0) for data in allocation_plan.values()]
            
            avg_diversity = float(np.mean(diversity_scores)) if diversity_scores else 0.0
            avg_quality = float(np.mean(quality_scores)) if quality_scores else 0.0
            avg_vector_score = float(np.mean(vector_scores)) if vector_scores else 0.0
            
            optimization_results = {
                "original_image_count": len(original_images),
                "allocated_image_count": total_allocated,
                "utilization_rate": total_allocated / len(original_images) if original_images else 0,
                "section_count": len(sections),
                "average_diversity_score": avg_diversity,
                "average_quality_score": avg_quality,
                "average_vector_score": avg_vector_score,
                "clip_used": self.clip_available,
                "vector_enhanced": True,
                "optimization_timestamp": time.time(),
                "allocation_details": allocation_plan
            }
            
            await self.logging_manager.log_agent_response(
                agent_name="ImageDiversityManager",
                agent_role="벡터 통합 이미지 다양성 관리 에이전트",
                task_description=f"벡터 기반 이미지 다양성 최적화: {len(original_images)}개 → {total_allocated}개 할당",
                response_data=optimization_results,
                metadata={
                    "utilization_rate": optimization_results['utilization_rate'],
                    "average_diversity": avg_diversity,
                    "average_quality": avg_quality,
                    "average_vector_score": avg_vector_score,
                    "vector_enhanced": True
                }
            )
            
            try:
                self.store_result(optimization_results)
            except Exception as session_error:
                self.logger.warning(f"세션 결과 저장 실패: {session_error}")
            
            self.logger.info(f"벡터 강화 다양성 최적화 결과 - 활용률: {optimization_results['utilization_rate']:.2%}, "
                        f"평균 다양성: {avg_diversity:.3f}, 평균 품질: {avg_quality:.3f}, 평균 벡터 점수: {avg_vector_score:.3f}")
            
        except Exception as e:
            self.logger.error(f"결과 로깅 실패: {e}")

    def get_optimization_statistics(self) -> Dict:
        """최적화 통계 반환"""
        return {
            "processed_hashes_count": len(self.processed_hashes),
            "cached_embeddings_count": len(self.image_embeddings_cache),
            "clip_available": self.clip_available,
            "similarity_threshold": self.similarity_threshold,
            "diversity_weight": self.diversity_weight,
            "vector_integrated": True
        }

    # 품질 평가 메트릭 메서드들 (기존 유지)
    def _calculate_sharpness(self, img_array: np.ndarray) -> float:
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            return min(laplacian_var / 1000.0, 1.0)
        except:
            return 0.5

    def _calculate_contrast(self, img_array: np.ndarray) -> float:
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            return min(gray.std() / 255.0, 1.0)
        except:
            return 0.5

    def _calculate_brightness(self, img_array: np.ndarray) -> float:
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            mean_brightness = gray.mean() / 255.0
            if 0.3 <= mean_brightness <= 0.7:
                return 1.0
            else:
                return max(1.0 - abs(mean_brightness - 0.5) * 2, 0.0)
        except:
            return 0.5

    def _calculate_composition_score(self, img_array: np.ndarray) -> float:
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            h, w = gray.shape
            
            third_points = [
                (w//3, h//3), (2*w//3, h//3),
                (w//3, 2*h//3), (2*w//3, 2*h//3)
            ]
            
            interest_scores = []
            for x, y in third_points:
                region = gray[max(0, y-20):min(h, y+20), max(0, x-20):min(w, x+20)]
                if region.size > 0:
                    interest_scores.append(region.std())
            
            avg_interest = np.mean(interest_scores) if interest_scores else 0
            return min(avg_interest / 50.0, 1.0)
        except:
            return 0.5
