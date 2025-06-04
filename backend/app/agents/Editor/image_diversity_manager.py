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
import torch
import open_clip
from utils.hybridlogging import get_hybrid_logger
from utils.session_isolation import SessionAwareMixin
from utils.logging_manager import LoggingManager

class ImageDiversityManager(SessionAwareMixin):
    """이미지 다양성 관리 및 중복 방지 전문 에이전트"""
    
    def __init__(self, similarity_threshold: int = 10, diversity_weight: float = 0.7):
        super().__init__()
        self.logger = get_hybrid_logger(self.__class__.__name__)
        self.similarity_threshold = similarity_threshold
        self.diversity_weight = diversity_weight
        self.logging_manager = LoggingManager()
        self.processed_hashes: Set[str] = set()
        self.image_clusters: Dict[str, List[Dict]] = {}
        self.image_embeddings_cache: Dict[str, np.ndarray] = {}
        
        self.__init_session_awareness__()        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # CLIP 모델 초기화
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='laion2b_s34b_b79k', device=self.device
            )
            self.clip_model.eval()  # 평가 모드로 설정
            self.clip_available = True
            self.logger.info("CLIP 모델 로드 성공 (open-clip-torch)")
        except Exception as e:
            self.logger.warning(f"CLIP 모델 로드 실패, 기본 해시 방식 사용: {e}")
            self.clip_available = False
        
        # 품질 평가 메트릭
        self.quality_metrics = {
            "sharpness": self._calculate_sharpness,
            "contrast": self._calculate_contrast,
            "brightness": self._calculate_brightness,
            "composition": self._calculate_composition_score
        }
        
        self.logger.info("ImageDiversityManager 초기화 완료")


    async def process_data(self, input_data):
        # 에이전트 작업 수행
        result = await self._do_work(input_data)
        
        # ✅ 응답 로그 저장
        await self.logging_manager.log_agent_response(
            agent_name=self.__class__.__name__,
            agent_role="에이전트 역할 설명",
            task_description="수행한 작업 설명",
            response_data=result, 
            metadata={"additional": "info"}
        )
        
        return result

    async def optimize_image_distribution(self, images: List[Dict], sections: List[Dict]) -> Dict[str, List[Dict]]:
        """전역 최적화 기반 이미지 분배 (메인 메서드)"""
        self.logger.info(f"이미지 다양성 최적화 시작: {len(images)}개 이미지, {len(sections)}개 섹션")
        
        try:
            # 1. 중복 이미지 제거
            unique_images = await self._remove_duplicate_images_async(images)
            
            # 2. 이미지 품질 평가
            quality_enhanced_images = await self._enhance_image_quality_scores(unique_images)
            
            # 3. CLIP 기반 이미지 클러스터링 (가능한 경우)
            if self.clip_available and len(quality_enhanced_images) > 5:
                clustered_images = await self._cluster_images_with_clip(quality_enhanced_images)
            else:
                clustered_images = {"default_cluster": quality_enhanced_images}
            
            # 4. 클러스터별 대표 이미지 선택
            representative_images = self._select_representative_images(clustered_images)
            
            # 5. 다양성을 고려한 섹션별 이미지 할당
            allocation_plan = await self._allocate_images_to_sections_with_diversity(
                representative_images, sections
            )
            
            # 6. 결과 로깅 및 저장
            await self._log_diversity_optimization_results(allocation_plan, images, sections)
            
            self.logger.info(f"이미지 다양성 최적화 완료: {len(allocation_plan)}개 섹션에 할당")
            return allocation_plan
            
        except Exception as e:
            self.logger.error(f"이미지 다양성 최적화 실패: {e}")
            # 폴백: 기본 순차 분배
            return self._fallback_sequential_distribution(images, sections)

    async def _remove_duplicate_images_async(self, images: List[Dict]) -> List[Dict]:
        """비동기 중복 이미지 제거 (Perceptual Hashing 사용)"""
        unique_images = []
        
        for image_data in images:
            try:
                image_url = image_data.get("image_url", "")
                image_name = image_data.get("image_name", "")
                
                if not image_url:
                    continue
                
                # Perceptual hash 계산
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
                # 오류 발생 시에도 이미지 포함 (안전성)
                unique_images.append(image_data)
        
        self.logger.info(f"중복 제거 완료: {len(images)} → {len(unique_images)}개")
        return unique_images

    async def _calculate_perceptual_hash_async(self, image_url: str) -> Optional[str]:
        """비동기 Perceptual Hash 계산"""
        try:
            # 이미지 로드 (비동기)
            loop = asyncio.get_event_loop()
            image_hash = await loop.run_in_executor(
                None, self._calculate_perceptual_hash_sync, image_url
            )
            return image_hash
        except Exception as e:
            self.logger.error(f"Perceptual hash 계산 실패: {e}")
            return None

    def _calculate_perceptual_hash_sync(self, image_url: str) -> str:
        """동기 Perceptual Hash 계산 (URL 지원)"""
        try:
            # ✅ URL인지 로컬 파일인지 확인
            if image_url.startswith(('http://', 'https://')):
                # URL에서 이미지 다운로드
                response = self.session.get(image_url, timeout=30)
                response.raise_for_status()
                
                # Content-Type 확인
                content_type = response.headers.get('content-type', '')
                if not content_type.startswith('image/'):
                    self.logger.warning(f"URL이 이미지가 아님: {content_type}")
                    return ""
                
                # BytesIO로 변환하여 PIL에서 처리
                image_data = BytesIO(response.content)
                with Image.open(image_data) as img:
                    # RGB로 변환 (RGBA나 다른 모드 처리)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    phash = imagehash.phash(img, hash_size=16)
                    return str(phash)
            else:
                # 로컬 파일 처리
                with Image.open(image_url) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    phash = imagehash.phash(img, hash_size=16)
                    return str(phash)
                    
        except Exception as e:
            self.logger.error(f"이미지 해시 계산 실패 {image_url}: {e}")
            return ""

    def _is_duplicate_or_similar(self, image_hash: str) -> bool:
        """중복 또는 유사 이미지 검사 (Hamming Distance 사용)"""
        try:
            current_hash = imagehash.hex_to_hash(image_hash)
            
            for existing_hash_str in self.processed_hashes:
                existing_hash = imagehash.hex_to_hash(existing_hash_str)
                # Hamming distance로 유사도 측정
                distance = current_hash - existing_hash
                
                if distance <= self.similarity_threshold:
                    return True
            
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
                    # 품질 점수 계산
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
        """동기 이미지 품질 평가 (URL 지원)"""
        try:
            # ✅ URL 처리 로직 추가
            if image_url.startswith(('http://', 'https://')):
                response = self.session.get(image_url, timeout=30)
                response.raise_for_status()
                
                image_data = BytesIO(response.content)
                with Image.open(image_data) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img_array = np.array(img)
            else:
                with Image.open(image_url) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img_array = np.array(img)
            
            quality_scores = {}
            for metric_name, metric_func in self.quality_metrics.items():
                try:
                    quality_scores[metric_name] = metric_func(img_array)
                except Exception as e:
                    self.logger.debug(f"품질 메트릭 {metric_name} 계산 실패: {e}")
                    quality_scores[metric_name] = 0.5
            
            # 종합 품질 점수 계산 (가중 평균)
            weights = {"sharpness": 0.3, "contrast": 0.3, "brightness": 0.2, "composition": 0.2}
            overall_score = sum(quality_scores.get(metric, 0.5) * weight 
                              for metric, weight in weights.items())
            quality_scores["overall"] = min(max(overall_score, 0.0), 1.0)
            
            return quality_scores
        except Exception as e:
            return {"overall": 0.3, "error": str(e)}

    def _calculate_sharpness(self, img_array: np.ndarray) -> float:
        """이미지 선명도 계산 (Laplacian variance)"""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            # 정규화 (0-1 범위)
            return min(laplacian_var / 1000.0, 1.0)
        except:
            return 0.5

    def _calculate_contrast(self, img_array: np.ndarray) -> float:
        """이미지 대비 계산"""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            return min(gray.std() / 255.0, 1.0)
        except:
            return 0.5

    def _calculate_brightness(self, img_array: np.ndarray) -> float:
        """이미지 밝기 적절성 계산"""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            mean_brightness = gray.mean() / 255.0
            # 0.3-0.7 범위가 최적, 이를 벗어나면 점수 감소
            if 0.3 <= mean_brightness <= 0.7:
                return 1.0
            else:
                return max(1.0 - abs(mean_brightness - 0.5) * 2, 0.0)
        except:
            return 0.5

    def _calculate_composition_score(self, img_array: np.ndarray) -> float:
        """이미지 구성 점수 계산 (Rule of Thirds 근사)"""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            h, w = gray.shape
            
            # Rule of thirds 포인트들
            third_points = [
                (w//3, h//3), (2*w//3, h//3),
                (w//3, 2*h//3), (2*w//3, 2*h//3)
            ]
            
            # 각 포인트 주변의 관심도 측정
            interest_scores = []
            for x, y in third_points:
                # 주변 영역의 표준편차 (텍스처/디테일 측정)
                region = gray[max(0, y-20):min(h, y+20), max(0, x-20):min(w, x+20)]
                if region.size > 0:
                    interest_scores.append(region.std())
            
            # 평균 관심도를 0-1 범위로 정규화
            avg_interest = np.mean(interest_scores) if interest_scores else 0
            return min(avg_interest / 50.0, 1.0)
        except:
            return 0.5

    async def _cluster_images_with_clip(self, images: List[Dict]) -> Dict[str, List[Dict]]:
        """CLIP 기반 이미지 클러스터링"""
        if not self.clip_available or len(images) < 3:
            return {"default_cluster": images}
        
        try:
            self.logger.info(f"CLIP 기반 클러스터링 시작: {len(images)}개 이미지")
            
            # CLIP 임베딩 생성
            embeddings = await self._generate_clip_embeddings(images)
            
            if embeddings is None or len(embeddings) == 0:
                return {"default_cluster": images}
            
            # DBSCAN 클러스터링 수행
            clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
            cluster_labels = clustering.fit_predict(embeddings)
            
            # 클러스터별 이미지 그룹화
            clusters = {}
            for i, label in enumerate(cluster_labels):
                cluster_key = f"cluster_{label}" if label != -1 else "outliers"
                
                if cluster_key not in clusters:
                    clusters[cluster_key] = []
                
                clusters[cluster_key].append(images[i])
            
            self.logger.info(f"클러스터링 완료: {len(clusters)}개 클러스터")
            return clusters
            
        except Exception as e:
            self.logger.error(f"CLIP 클러스터링 실패: {e}")
            return {"default_cluster": images}

    async def _generate_clip_embeddings(self, images: List[Dict]) -> Optional[np.ndarray]:
        """CLIP 임베딩 생성 (URL 지원)"""
        try:
            embeddings = []
            
            for image_data in images:
                image_url = image_data.get("image_url", "")
                
                # 캐시 확인
                if image_url in self.image_embeddings_cache:
                    embeddings.append(self.image_embeddings_cache[image_url])
                    continue
                
                try:
                    # ✅ URL 처리 개선
                    if image_url.startswith(('http://', 'https://')):
                        response = self.session.get(image_url, timeout=30)
                        response.raise_for_status()
                        
                        image_data_bytes = BytesIO(response.content)
                        image = Image.open(image_data_bytes)
                    else:
                        image = Image.open(image_url)
                    
                    # RGB 변환
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        image_features = self.clip_model.encode_image(image_input)
                        # L2 정규화
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        embedding = image_features.cpu().numpy().flatten()
                        
                        # 캐시에 저장
                        self.image_embeddings_cache[image_url] = embedding
                        embeddings.append(embedding)
                        
                except Exception as e:
                    self.logger.error(f"CLIP 임베딩 생성 실패 {image_url}: {e}")
                    # 기본 임베딩 추가 (512차원 영벡터)
                    embeddings.append(np.zeros(512))
            
            return np.array(embeddings) if embeddings else None
            
        except Exception as e:
            self.logger.error(f"CLIP 임베딩 생성 전체 실패: {e}")
            return None

    def _select_representative_images(self, clusters: Dict[str, List[Dict]]) -> List[Dict]:
        """
        클러스터별 대표 이미지 선정 (활용률 극대화, 중복 제거)
        - 각 클러스터에서 품질 기준을 만족하는 이미지를 모두 선택
        - 너무 유사한 이미지는 해시 중복으로 제거
        """
        representative_images = []
        temp_hashes = set()
        for cluster_name, cluster_images in clusters.items():
            # 품질 기준(예: 0.4 이상) 만족하는 모든 이미지 선택
            for img in cluster_images:
                img_hash = img.get("perceptual_hash")
                quality = img.get("overall_quality", 0.5)
                if quality >= 0.4 and img_hash and img_hash not in temp_hashes:
                    representative_images.append(img)
                    temp_hashes.add(img_hash)
                elif not img_hash:  # 해시가 없으면 일단 포함
                    representative_images.append(img)
        self.logger.info(f"대표 이미지 선택 완료(활용률 극대화): {len(representative_images)}개")
        return representative_images

    async def _allocate_images_to_sections_with_diversity(self, images: List[Dict], sections: List[Dict]) -> Dict[str, List[Dict]]:
        """
        섹션별 의미적 유사도 기반 이미지 할당 (중복 없이, 최대한 많은 이미지 활용)
        - 각 섹션의 텍스트와 모든 이미지 간의 CLIP 유사도 계산
        - 각 이미지는 한 섹션에만 할당 (중복 방지)
        - 섹션당 할당 이미지 개수는 min(3, 남은 이미지/남은 섹션)로 동적 조정
        """
        total_images = len(images)
        total_sections = len(sections)
        if total_images == 0 or total_sections == 0:
            return {f"section_{i}": {"images": [], "count": 0, "section_title": sections[i].get("title", f"섹션 {i+1}"), "diversity_score": 0.0, "avg_quality": 0.0} for i in range(total_sections)}

        # CLIP 임베딩 생성
        if self.clip_available:
            image_embeddings = await self._generate_clip_embeddings(images)
            section_texts = [sec.get("title", "") + " " + sec.get("content", "")[:200] for sec in sections]
            section_embeddings = await self._generate_clip_text_embeddings(section_texts)
            similarity_matrix = cosine_similarity(section_embeddings, image_embeddings)
        else:
            similarity_matrix = None

        allocation_plan = {}
        assigned_image_indices = set()
        images_per_section = max(1, total_images // total_sections)
        images_left = total_images

        for i, section in enumerate(sections):
            allocation_count = min(3, images_per_section, images_left - (total_sections - i - 1))
            if allocation_count <= 0:
                allocation_count = 1

            # 유사도 기반 최적 이미지 선택
            if similarity_matrix is not None:
                sim_scores = list(enumerate(similarity_matrix[i]))
                # 아직 할당되지 않은 이미지 중 유사도 높은 순으로 정렬
                sim_scores = [s for s in sim_scores if s[0] not in assigned_image_indices]
                sim_scores.sort(key=lambda x: x[1], reverse=True)
                selected_indices = [idx for idx, _ in sim_scores[:allocation_count]]
            else:
                # CLIP 사용 불가 시 품질 기준
                quality_scores = [(idx, img.get("overall_quality", 0.5)) for idx, img in enumerate(images) if idx not in assigned_image_indices]
                quality_scores.sort(key=lambda x: x[1], reverse=True)
                selected_indices = [idx for idx, _ in quality_scores[:allocation_count]]

            selected_images = [images[idx] for idx in selected_indices]
            for idx in selected_indices:
                assigned_image_indices.add(idx)

            allocation_plan[f"section_{i}"] = {
                "images": selected_images,
                "count": len(selected_images),
                "section_title": section.get("title", f"섹션 {i+1}"),
                "diversity_score": self._calculate_section_diversity(selected_images, image_embeddings[selected_indices] if self.clip_available else None),
                "avg_quality": float(np.mean([img.get("overall_quality", 0.5) for img in selected_images])) if selected_images else 0.0
            }
            images_left -= len(selected_images)

        return allocation_plan

    async def _generate_clip_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        CLIP 텍스트 임베딩 생성
        """
        import torch
        with torch.no_grad():
            text_tokens = open_clip.tokenize(texts).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy()

    def _select_diverse_images_with_clip(self, images: List[Dict], 
                                       embeddings: np.ndarray, 
                                       target_count: int) -> List[Dict]:
        """CLIP 임베딩 기반 다양성 고려 이미지 선택"""
        
        if len(images) <= target_count:
            return images
        
        selected_images = []
        selected_embeddings = []
        remaining_indices = list(range(len(images)))
        
        # 첫 번째 이미지는 품질이 가장 높은 것 선택
        quality_scores = [img.get("overall_quality", 0.3) for img in images]
        first_idx = np.argmax(quality_scores)
        
        selected_images.append(images[first_idx])
        selected_embeddings.append(embeddings[first_idx])
        remaining_indices.remove(first_idx)
        
        # 나머지 이미지들은 다양성 점수 기반 선택
        while len(selected_images) < target_count and remaining_indices:
            best_score = -1
            best_idx = None
            
            for idx in remaining_indices:
                # 다양성 점수 계산 (선택된 이미지들과의 최소 거리)
                diversity_score = self._calculate_diversity_score(
                    embeddings[idx], selected_embeddings
                )
                
                # 품질 점수
                quality_score = images[idx].get("overall_quality", 0.3)
                
                # 종합 점수 = 다양성 * weight + 품질 * (1-weight)
                total_score = (diversity_score * self.diversity_weight + 
                             quality_score * (1 - self.diversity_weight))
                
                if total_score > best_score:
                    best_score = total_score
                    best_idx = idx
            
            if best_idx is not None:
                selected_images.append(images[best_idx])
                selected_embeddings.append(embeddings[best_idx])
                remaining_indices.remove(best_idx)
        
        return selected_images

    def _calculate_diversity_score(self, candidate_embedding: np.ndarray, 
                                 selected_embeddings: List[np.ndarray]) -> float:
        """다양성 점수 계산 (코사인 거리 기반)"""
        if not selected_embeddings:
            return 1.0
        
        # 선택된 이미지들과의 코사인 유사도 계산
        similarities = []
        for selected_emb in selected_embeddings:
            similarity = cosine_similarity([candidate_embedding], [selected_emb])[0][0]
            similarities.append(similarity)
        
        # 최대 유사도를 다양성 점수로 변환 (낮을수록 다양함)
        max_similarity = max(similarities)
        diversity_score = 1.0 - max_similarity
        
        return max(diversity_score, 0.0)

    def _select_images_by_quality(self, images: List[Dict], target_count: int) -> List[Dict]:
        """품질 기반 이미지 선택 (CLIP 없을 때 폴백)"""
        if len(images) <= target_count:
            return images
        
        # 품질 점수 기준으로 정렬 후 상위 선택
        sorted_images = sorted(images, 
                             key=lambda x: x.get("overall_quality", 0.3), 
                             reverse=True)
        
        return sorted_images[:target_count]

    def _calculate_section_diversity(self, images: List[Dict], 
                                   embeddings: Optional[np.ndarray]) -> float:
        """섹션 내 이미지 다양성 점수 계산"""
        if len(images) <= 1:
            return 1.0
        
        if embeddings is None or not self.clip_available:
            # CLIP 없을 때는 품질 분산으로 대체
            quality_scores = [img.get("overall_quality", 0.5) for img in images]
            return float(np.std(quality_scores)) if quality_scores else 0.5
        
        try:
            # 이미지별 임베딩 추출
            section_embeddings = []
            for image in images:
                image_url = image.get("image_url", "")
                if image_url in self.image_embeddings_cache:
                    section_embeddings.append(self.image_embeddings_cache[image_url])
            
            if len(section_embeddings) < 2:
                return 0.5
            
            # 모든 이미지 쌍 간의 평균 코사인 거리 계산
            total_distance = 0
            pair_count = 0
            
            for i in range(len(section_embeddings)):
                for j in range(i + 1, len(section_embeddings)):
                    similarity = cosine_similarity([section_embeddings[i]], [section_embeddings[j]])[0][0]
                    distance = 1.0 - similarity  # 코사인 거리
                    total_distance += distance
                    pair_count += 1
            
            return total_distance / pair_count if pair_count > 0 else 0.5
            
        except Exception as e:
            self.logger.error(f"다양성 점수 계산 실패: {e}")
            return 0.5

    def _fallback_sequential_distribution(self, images: List[Dict], 
                                        sections: List[Dict]) -> Dict[str, List[Dict]]:
        """폴백: 순차적 이미지 분배"""
        self.logger.warning("폴백 모드: 순차적 이미지 분배 사용")
        
        total_images = len(images)
        total_sections = len(sections)
        
        if total_images == 0:
            return {f"section_{i}": [] for i in range(total_sections)}
        
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
                "fallback_used": True
            }
            
            current_idx = end_idx
        
        return allocation_plan

    async def _log_diversity_optimization_results(self, allocation_plan: Dict, 
                                            original_images: List[Dict], 
                                            sections: List[Dict]) -> None:
        """다양성 최적화 결과 로깅 (새로운 방식 적용)"""
        try:
            # 통계 계산
            total_allocated = sum(data["count"] for data in allocation_plan.values())
            diversity_scores = [data["diversity_score"] for data in allocation_plan.values()]
            quality_scores = [data["avg_quality"] for data in allocation_plan.values()]
            
            avg_diversity = float(np.mean(diversity_scores)) if diversity_scores else 0.0
            avg_quality = float(np.mean(quality_scores)) if quality_scores else 0.0
            
            optimization_results = {
                "original_image_count": len(original_images),
                "allocated_image_count": total_allocated,
                "utilization_rate": total_allocated / len(original_images) if original_images else 0,
                "section_count": len(sections),
                "average_diversity_score": avg_diversity,
                "average_quality_score": avg_quality,
                "clip_used": self.clip_available,
                "optimization_timestamp": time.time(),
                "allocation_details": allocation_plan
            }
            
            # ✅ LoggingManager 사용
            from utils.logging_manager import LoggingManager
            logging_manager = LoggingManager()
            
            await logging_manager.log_agent_response(
                agent_name="ImageDiversityManager",
                agent_role="이미지 다양성 관리 및 중복 방지 전문 에이전트",
                task_description=f"이미지 다양성 최적화: {len(original_images)}개 → {total_allocated}개 할당",
                response_data=optimization_results,  # ✅ 실제 최적화 결과만 저장
                metadata={
                    "utilization_rate": optimization_results['utilization_rate'],
                    "average_diversity": avg_diversity,
                    "average_quality": avg_quality,
                    "clip_available": self.clip_available
                }
            )
            
            # ✅ 안전한 세션 결과 저장
            try:
                self.store_result(optimization_results)
            except Exception as session_error:
                self.logger.warning(f"세션 결과 저장 실패: {session_error}")
            
            self.logger.info(f"다양성 최적화 결과 - 활용률: {optimization_results['utilization_rate']:.2%}, "
                        f"평균 다양성: {avg_diversity:.3f}, 평균 품질: {avg_quality:.3f}")
            
        except Exception as e:
            self.logger.error(f"결과 로깅 실패: {e}")

    def get_optimization_statistics(self) -> Dict:
        """최적화 통계 반환"""
        return {
            "processed_hashes_count": len(self.processed_hashes),
            "cached_embeddings_count": len(self.image_embeddings_cache),
            "clip_available": self.clip_available,
            "similarity_threshold": self.similarity_threshold,
            "diversity_weight": self.diversity_weight
        }
