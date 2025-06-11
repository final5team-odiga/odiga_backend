import asyncio
import numpy as np
import time
import os
from typing import List, Dict, Set, Optional
from PIL import Image
import requests
from io import BytesIO
import imagehash
import cv2
from dotenv import load_dotenv

from azure.storage.blob import BlobServiceClient

from sklearn.cluster import DBSCAN
from ...utils.log.hybridlogging import HybridLogger
from ...utils.isolation.session_isolation import SessionAwareMixin
from ...utils.log.logging_manager import LoggingManager
from ...utils.data.pdf_vector_manager import PDFVectorManager
from ...utils.isolation.ai_search_isolation import AISearchIsolationManager
import onnxruntime as ort

class ImageDiversityManager(SessionAwareMixin):
    """이미지 다양성 관리 및 중복 방지 전문 에이전트 - 블롭 스토리지 통합 버전"""
    
    def __init__(self, vector_manager: PDFVectorManager, logger: 'HybridLogger', 
                 similarity_threshold: int = 40, diversity_weight: float = 0.3):
        super().__init__()
        self.logger = logger
        self.logging_manager = LoggingManager(self.logger)
        
        # ✅ 블롭 스토리지 매니저 초기화
        self._initialize_blob_storage()
        
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
        
        self.logger.info("ImageDiversityManager 초기화 완료 (블롭 스토리지 통합 버전)")

    def _initialize_blob_storage(self):
        """✅ 블롭 스토리지 클라이언트 초기화"""
        try:
            load_dotenv()
            
            connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            self.container_name = os.getenv("AZURE_STORAGE_CONTAINER")
            
            if connection_string and self.container_name:
                self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                self.container_client = self.blob_service_client.get_container_client(self.container_name)
                self.blob_storage_available = True
                self.logger.info("✅ 블롭 스토리지 클라이언트 초기화 성공")
            else:
                self.blob_storage_available = False
                self.logger.warning("블롭 스토리지 연결 정보가 없습니다")
                
        except Exception as e:
            self.logger.error(f"블롭 스토리지 초기화 실패: {e}")
            self.blob_storage_available = False

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
        """✅ 모든 섹션 이미지 배치 보장 최적화"""
        self.logger.info(f"전체 섹션 이미지 배치 최적화 시작: {len(images)}개 이미지, {len(sections)}개 섹션")
        
        try:
            # ✅ 1. 실제 섹션 수 정확히 계산 (하위 섹션 포함)
            actual_sections = self._calculate_actual_sections(sections)
            total_sections = len(actual_sections)
            
            self.logger.info(f"실제 섹션 수: {total_sections}개 (하위 섹션 포함)")
            
            # 2. 강화된 중복 이미지 제거
            unique_images = await self._enhanced_duplicate_removal(images)
            
            # ✅ 3. 이미지 부족 시 확장 처리
            if len(unique_images) < total_sections:
                expanded_images = self._expand_image_pool(unique_images, total_sections)
                self.logger.info(f"이미지 풀 확장: {len(unique_images)} → {len(expanded_images)}개")
            else:
                expanded_images = unique_images
            
            # 4. 이미지 품질 평가
            quality_enhanced_images = await self._enhance_image_quality_scores(expanded_images)
            
            # 5. 벡터 검색 기반 이미지 의미 패턴 수집
            if unified_patterns:
                semantic_patterns = await self._collect_image_semantic_patterns(
                    quality_enhanced_images, unified_patterns
                )
            else:
                semantic_patterns = {}
            
            # 6. CLIP 기반 이미지 클러스터링 (가능한 경우)
            if self.clip_available and len(quality_enhanced_images) > 5:
                clustered_images = await self._cluster_images_with_clip_and_vectors(
                    quality_enhanced_images, semantic_patterns
                )
            else:
                clustered_images = {"default_cluster": quality_enhanced_images}
            
            # 7. 벡터 패턴 기반 대표 이미지 선택 (중복 방지 강화)
            representative_images = self._select_representative_images_with_enhanced_deduplication(
                clustered_images, semantic_patterns
            )
            
            # ✅ 8. 모든 섹션에 균등 배치 보장
            allocation_plan = await self._allocate_images_to_all_sections(
                representative_images, actual_sections, semantic_patterns
            )
            
            # 9. 결과 로깅
            await self._log_diversity_optimization_results(allocation_plan, images, actual_sections)
            
            self.logger.info(f"전체 섹션 이미지 배치 완료: {len(allocation_plan)}개 섹션에 할당")
            
            return {
                "allocation_plan": allocation_plan,
                "allocation_details": allocation_plan,
                "embeddings_cache": self.image_embeddings_cache,
                "semantic_patterns": semantic_patterns,
                "vector_enhanced": True,
                "total_sections": total_sections,
                "all_sections_covered": True,
                "deduplication_applied": True
            }
            
        except Exception as e:
            self.logger.error(f"전체 섹션 이미지 배치 실패: {e}")
            # 기본 균등 분배로 폴백
            return self._ensure_all_sections_have_images(images, sections)

    def _calculate_actual_sections(self, sections: List[Dict]) -> List[Dict]:
        """✅ 하위 섹션을 포함한 실제 섹션 리스트 계산"""
        actual_sections = []
        
        for i, section in enumerate(sections):
            sub_sections = section.get("sub_sections", [])
            
            if sub_sections:
                # 하위 섹션이 있는 경우 각각을 개별 섹션으로 처리
                for j, sub_section in enumerate(sub_sections):
                    actual_section = {
                        "section_index": len(actual_sections),
                        "original_index": i,
                        "sub_index": j,
                        "title": f"{section.get('title', '')}: {sub_section.get('title', '')}",
                        "content": sub_section.get("body", ""),
                        "is_subsection": True,
                        "parent_title": section.get("title", "")
                    }
                    actual_sections.append(actual_section)
            else:
                # 단일 섹션인 경우
                actual_section = {
                    "section_index": len(actual_sections),
                    "original_index": i,
                    "sub_index": None,
                    "title": section.get("title", f"섹션 {i+1}"),
                    "content": section.get("content", section.get("body", "")),
                    "is_subsection": False,
                    "parent_title": ""
                }
                actual_sections.append(actual_section)
        
        return actual_sections

    async def _enhanced_duplicate_removal(self, images: List[Dict]) -> List[Dict]:
        """✅ 강화된 중복 제거 (URL + Hash + Content 기반)"""
        unique_images = []
        seen_urls = set()
        seen_hashes = set()
        processed_content_hashes = set()
        
        for image_data in images:
            try:
                image_url = image_data.get("image_url", "")
                image_name = image_data.get("image_name", "")
                
                if not image_url:
                    continue
                
                # ✅ 1. URL 기반 1차 중복 검사
                if image_url in seen_urls:
                    self.logger.debug(f"URL 중복 제거: {image_name}")
                    continue
                
                # ✅ 2. Perceptual Hash 기반 2차 중복 검사
                image_hash = await self._calculate_perceptual_hash_async(image_url)
                
                if image_hash:
                    if image_hash in seen_hashes or self._is_duplicate_or_similar(image_hash):
                        self.logger.debug(f"해시 중복 제거: {image_name}")
                        continue
                    
                    # ✅ 3. 콘텐츠 기반 3차 중복 검사
                    content_hash = self._generate_content_hash(image_data)
                    if content_hash in processed_content_hashes:
                        self.logger.debug(f"콘텐츠 중복 제거: {image_name}")
                        continue
                    
                    # 모든 검사를 통과한 경우 추가
                    seen_urls.add(image_url)
                    seen_hashes.add(image_hash)
                    processed_content_hashes.add(content_hash)
                    self.processed_hashes.add(image_hash)
                    
                    image_data["perceptual_hash"] = image_hash
                    image_data["content_hash"] = content_hash
                    unique_images.append(image_data)
                    
                    self.logger.debug(f"고유 이미지 추가: {image_name}")
                else:
                    # 해시 계산 실패 시 URL만으로 중복 검사
                    seen_urls.add(image_url)
                    unique_images.append(image_data)
                    
            except Exception as e:
                self.logger.error(f"강화된 중복 검사 실패 {image_data.get('image_name', 'Unknown')}: {e}")
                # 오류 발생 시에도 이미지 포함 (안전장치)
                if image_data not in unique_images:
                    unique_images.append(image_data)
        
        self.logger.info(f"강화된 중복 제거 완료: {len(images)} → {len(unique_images)}개")
        return unique_images

    def _generate_content_hash(self, image_data: Dict) -> str:
        """이미지 메타데이터 기반 콘텐츠 해시 생성"""
        try:
            # 이미지의 주요 메타데이터를 조합하여 해시 생성
            content_parts = [
                image_data.get("city", ""),
                image_data.get("country", ""),
                image_data.get("location", ""),
                str(image_data.get("width", 0)),
                str(image_data.get("height", 0))
            ]
            
            content_string = "|".join(content_parts)
            
            import hashlib
            return hashlib.md5(content_string.encode('utf-8')).hexdigest()
            
        except Exception as e:
            self.logger.error(f"콘텐츠 해시 생성 실패: {e}")
            return ""

    def _expand_image_pool(self, unique_images: List[Dict], required_count: int) -> List[Dict]:
        """✅ 이미지 부족 시 이미지 풀 확장"""
        if len(unique_images) >= required_count:
            return unique_images
        
        expanded_images = unique_images.copy()
        
        # 기존 이미지를 순환하여 필요한 수만큼 확장
        while len(expanded_images) < required_count:
            for img in unique_images:
                if len(expanded_images) >= required_count:
                    break
                
                # 이미지 복사본 생성 (URL은 유지, 메타데이터 약간 변경)
                expanded_img = img.copy()
                expanded_img["expanded_copy"] = True
                expanded_img["original_index"] = unique_images.index(img)
                
                expanded_images.append(expanded_img)
        
        return expanded_images

    async def _allocate_images_to_all_sections(self, images: List[Dict], 
                                             actual_sections: List[Dict], 
                                             semantic_patterns: Dict) -> Dict:
        """✅ 모든 섹션에 이미지 배치 보장"""
        total_images = len(images)
        total_sections = len(actual_sections)
        
        if total_sections == 0:
            return {}
        
        # ✅ 각 섹션에 최소 1개씩 배치 보장
        base_allocation = max(1, total_images // total_sections)
        remainder = total_images % total_sections
        
        allocation_plan = {}
        current_image_index = 0
        
        for i, section in enumerate(actual_sections):
            # 할당량 계산 (나머지 이미지 분배)
            allocation_count = base_allocation
            if i < remainder:
                allocation_count += 1
            
            # 이미지 할당
            end_index = min(current_image_index + allocation_count, total_images)
            allocated_images = images[current_image_index:end_index]
            
            # 이미지가 부족한 경우 첫 번째 이미지로 채우기
            while len(allocated_images) < 1 and images:
                allocated_images.append(images[0])
            
            section_key = f"section_{section['section_index']}"
            
            allocation_plan[section_key] = {
                "images": allocated_images,
                "count": len(allocated_images),
                "section_title": section["title"],
                "diversity_score": 0.7,  # 기본 다양성 점수
                "avg_quality": 0.75,
                "vector_score": 0.0,
                "is_subsection": section["is_subsection"],
                "guaranteed_allocation": True
            }
            
            current_image_index = end_index
            
            self.logger.info(f"섹션 {section_key} 이미지 할당: {len(allocated_images)}개")
        
        return allocation_plan

    # ✅ 블롭 스토리지 지원 메서드들
    def _is_blob_storage_url(self, image_url: str) -> bool:
        """블롭 스토리지 URL인지 확인"""
        return "blob.core.windows.net" in image_url

    def _extract_blob_name_from_url(self, image_url: str) -> str:
        """URL에서 블롭 이름 추출"""
        try:
            parts = image_url.split('/')
            if len(parts) >= 5:
                container_index = parts.index(self.container_name) if self.container_name in parts else 4
                blob_name = '/'.join(parts[container_index + 1:])
                return blob_name
            return ""
        except Exception as e:
            self.logger.error(f"블롭 이름 추출 실패 {image_url}: {e}")
            return ""

    def _download_blob_image(self, image_url: str) -> Optional[BytesIO]:
        """✅ 블롭 스토리지에서 이미지 다운로드"""
        try:
            if not self.blob_storage_available:
                return None
                
            blob_name = self._extract_blob_name_from_url(image_url)
            if not blob_name:
                return None
            
            blob_client = self.container_client.get_blob_client(blob_name)
            download_stream = blob_client.download_blob()
            image_data = BytesIO(download_stream.readall())
            
            self.logger.debug(f"블롭 이미지 다운로드 성공: {blob_name}")
            return image_data
            
        except Exception as e:
            self.logger.error(f"블롭 이미지 다운로드 실패 {image_url}: {e}")
            return None

    def _download_image_http_fallback(self, image_url: str) -> Optional[BytesIO]:
        """HTTP 요청으로 이미지 다운로드 (폴백)"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'image/*,*/*;q=0.8',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive'
            }
            
            response = self.session.get(
                image_url, 
                timeout=30, 
                headers=headers,
                allow_redirects=True,
                verify=False
            )
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                self.logger.warning(f"URL이 이미지가 아님: {content_type}")
                return None
            
            return BytesIO(response.content)
            
        except Exception as e:
            self.logger.error(f"HTTP 이미지 다운로드 실패 {image_url}: {e}")
            return None

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
        """✅ 블롭 스토리지 지원 동기 Perceptual Hash 계산"""
        try:
            image_data = None
            
            # ✅ 블롭 스토리지 URL인 경우 블롭 클라이언트 사용
            if self._is_blob_storage_url(image_url):
                image_data = self._download_blob_image(image_url)
                if image_data is None:
                    self.logger.warning(f"블롭 이미지 다운로드 실패, HTTP 요청으로 폴백: {image_url}")
                    image_data = self._download_image_http_fallback(image_url)
            else:
                # 일반 HTTP URL인 경우
                if image_url.startswith(('http://', 'https://')):
                    image_data = self._download_image_http_fallback(image_url)
                else:
                    # 로컬 파일인 경우
                    with Image.open(image_url) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        phash = imagehash.phash(img, hash_size=16)
                        return str(phash)
            
            # 이미지 데이터 처리
            if image_data:
                with Image.open(image_data) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    phash = imagehash.phash(img, hash_size=16)
                    return str(phash)
            else:
                # ✅ 다운로드 실패 시에도 기본 해시 반환 (배치 보장)
                return f"fallback_hash_{hash(image_url)}"
                
        except Exception as e:
            self.logger.error(f"이미지 해시 계산 실패 {image_url}: {e}")
            # ✅ 실패해도 기본 해시 반환 (배치 보장)
            return f"error_hash_{hash(image_url)}"

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
        """✅ 이미지 품질 점수 향상 (접근 실패 시에도 처리)"""
        enhanced_images = []
        
        for image_data in images:
            try:
                image_url = image_data.get("image_url", "")
                
                if image_url:
                    # ✅ 접근 가능 여부와 관계없이 기본 품질 점수 부여
                    try:
                        quality_scores = await self._assess_image_quality_async(image_url)
                    except:
                        # 접근 실패 시 기본 점수
                        quality_scores = {
                            "overall": 0.6,
                            "sharpness": 0.6,
                            "contrast": 0.6,
                            "brightness": 0.6,
                            "composition": 0.6,
                            "note": "Default score due to access failure"
                        }
                    
                    image_data["quality_scores"] = quality_scores
                    image_data["overall_quality"] = quality_scores.get("overall", 0.6)
                else:
                    image_data["quality_scores"] = {"overall": 0.4}
                    image_data["overall_quality"] = 0.4
                
                enhanced_images.append(image_data)
                
            except Exception as e:
                self.logger.error(f"품질 점수 계산 실패: {e}")
                # ✅ 실패해도 이미지 포함 (배치 보장)
                image_data["quality_scores"] = {"overall": 0.4}
                image_data["overall_quality"] = 0.4
                enhanced_images.append(image_data)
        
        return enhanced_images

    async def _assess_image_quality_async(self, image_url: str) -> Dict[str, float]:
        """✅ 블롭 스토리지 지원 비동기 이미지 품질 평가"""
        try:
            loop = asyncio.get_event_loop()
            quality_scores = await loop.run_in_executor(
                None, self._assess_image_quality_sync, image_url
            )
            return quality_scores
        except Exception as e:
            self.logger.error(f"이미지 품질 평가 실패: {e}")
            return {"overall": 0.6, "error": str(e)}

    def _assess_image_quality_sync(self, image_url: str) -> Dict[str, float]:
        """✅ 블롭 스토리지 지원 동기 이미지 품질 평가"""
        try:
            image_data = None
            
            # ✅ 블롭 스토리지 URL인 경우
            if self._is_blob_storage_url(image_url):
                image_data = self._download_blob_image(image_url)
                if image_data is None:
                    # 폴백: HTTP 요청
                    image_data = self._download_image_http_fallback(image_url)
            else:
                # 일반 URL인 경우
                if image_url.startswith(('http://', 'https://')):
                    image_data = self._download_image_http_fallback(image_url)
                else:
                    # 로컬 파일
                    with Image.open(image_url) as img:
                        img_array = np.array(img)
                        return self._calculate_quality_metrics(img_array)
            
            # 이미지 데이터 품질 평가
            if image_data:
                with Image.open(image_data) as img:
                    img_array = np.array(img)
                    return self._calculate_quality_metrics(img_array)
            else:
                # 접근 실패 시 기본 점수
                return {
                    "overall": 0.6,
                    "sharpness": 0.6,
                    "contrast": 0.6,
                    "brightness": 0.6,
                    "composition": 0.6,
                    "note": "Default score due to blob access failure"
                }
                
        except Exception as e:
            self.logger.error(f"블롭 품질 평가 실패 {image_url}: {e}")
            return {
                "overall": 0.5,
                "sharpness": 0.5,
                "contrast": 0.5,
                "brightness": 0.5,
                "composition": 0.5,
                "note": f"Error: {str(e)}"
            }

    def _calculate_quality_metrics(self, img_array: np.ndarray) -> Dict[str, float]:
        """이미지 배열에서 품질 메트릭 계산"""
        try:
            quality_scores = {}
            
            for metric_name, metric_func in self.quality_metrics.items():
                quality_scores[metric_name] = metric_func(img_array)
            
            # 전체 품질 점수 계산
            overall_quality = np.mean(list(quality_scores.values()))
            quality_scores["overall"] = overall_quality
            
            return quality_scores
            
        except Exception as e:
            self.logger.error(f"품질 메트릭 계산 실패: {e}")
            return {
                "overall": 0.5,
                "sharpness": 0.5,
                "contrast": 0.5,
                "brightness": 0.5,
                "composition": 0.5,
                "note": "Calculation failed"
            }

    # ✅ 벡터 패턴 관련 메서드들
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

    # ✅ CLIP 관련 메서드들
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

    async def _generate_clip_embeddings(self, images: List[Dict]) -> Optional[np.ndarray]:
        """✅ 블롭 스토리지 지원 CLIP 임베딩 생성"""
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
                    pil_image = None
                    
                    # ✅ 블롭 스토리지 URL인 경우 블롭 클라이언트 사용
                    if self._is_blob_storage_url(image_url):
                        image_bytes = self._download_blob_image(image_url)
                        if image_bytes:
                            pil_image = Image.open(image_bytes)
                        else:
                            # 폴백: HTTP 요청
                            image_bytes = self._download_image_http_fallback(image_url)
                            if image_bytes:
                                pil_image = Image.open(image_bytes)
                    else:
                        # 일반 HTTP URL인 경우
                        if image_url.startswith(('http://', 'https://')):
                            image_bytes = self._download_image_http_fallback(image_url)
                            if image_bytes:
                                pil_image = Image.open(image_bytes)
                        else:
                            pil_image = Image.open(image_url)
                    
                    if pil_image:
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
                    else:
                        # 이미지 로드 실패 시 기본 임베딩
                        embeddings.append(np.zeros(512))
                        
                except Exception as e:
                    self.logger.error(f"블롭 기반 임베딩 생성 실패 {image_url}: {e}")
                    embeddings.append(np.zeros(512))
            
            return np.array(embeddings) if embeddings else None
            
        except Exception as e:
            self.logger.error(f"CLIP 임베딩 생성 전체 실패: {e}")
            return None

    def _select_representative_images_with_enhanced_deduplication(self, clusters: Dict[str, List[Dict]], 
                                                               semantic_patterns: Dict) -> List[Dict]:
        """✅ 강화된 중복 방지 대표 이미지 선택"""
        representative_images = []
        global_seen_hashes = set()
        global_seen_urls = set()
        global_seen_content = set()
        
        for cluster_name, cluster_images in clusters.items():
            scored_images = []
            
            for img in cluster_images:
                img_hash = img.get("perceptual_hash", "")
                img_url = img.get("image_url", "")
                content_hash = img.get("content_hash", "")
                quality = img.get("overall_quality", 0.5)
                
                # ✅ 전역 중복 검사
                if (img_hash and img_hash in global_seen_hashes) or \
                   (img_url and img_url in global_seen_urls) or \
                   (content_hash and content_hash in global_seen_content):
                    continue
                
                # 벡터 패턴 점수 추가
                image_index = cluster_images.index(img)
                pattern_key = f"image_{image_index}"
                vector_score = semantic_patterns.get(pattern_key, {}).get("semantic_score", 0.0)
                
                # 종합 점수 계산
                combined_score = (quality * 0.6) + (vector_score * 0.4)
                
                scored_images.append((img, combined_score, img_hash, img_url, content_hash))
            
            # 점수순으로 정렬
            scored_images.sort(key=lambda x: x[1], reverse=True)
            
            # 상위 이미지들 선택 (전역 중복 제거)
            for img, score, img_hash, img_url, content_hash in scored_images:
                if score >= 0.3:
                    representative_images.append(img)
                    
                    # 전역 중복 방지 세트에 추가
                    if img_hash:
                        global_seen_hashes.add(img_hash)
                    if img_url:
                        global_seen_urls.add(img_url)
                    if content_hash:
                        global_seen_content.add(content_hash)
        
        self.logger.info(f"강화된 중복 방지 대표 이미지 선택 완료: {len(representative_images)}개")
        return representative_images

    def _ensure_all_sections_have_images(self, images: List[Dict], sections: List[Dict]) -> Dict:
        """✅ 모든 섹션에 이미지 배치 보장 (기본 방식)"""
        actual_sections = self._calculate_actual_sections(sections)
        total_sections = len(actual_sections)
        
        if not images or total_sections == 0:
            return {}
        
        allocation_plan = {}
        
        for i, section in enumerate(actual_sections):
            # 순환 방식으로 이미지 할당
            image_index = i % len(images)
            allocated_image = images[image_index]
            
            section_key = f"section_{section['section_index']}"
            
            allocation_plan[section_key] = {
                "images": [allocated_image],
                "count": 1,
                "section_title": section["title"],
                "diversity_score": 0.5,
                "avg_quality": 0.5,
                "vector_score": 0.0,
                "is_subsection": section["is_subsection"],
                "fallback_allocation": True
            }
        
        self.logger.info(f"기본 방식으로 {total_sections}개 섹션에 이미지 할당 완료")
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
            "vector_integrated": True,
            "blob_storage_available": self.blob_storage_available
        }

    # ✅ 품질 평가 메트릭 메서드들
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
