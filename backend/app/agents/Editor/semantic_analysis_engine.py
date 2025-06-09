import asyncio
import aiohttp
from PIL import Image
from io import BytesIO
import numpy as np
import open_clip
import time
import torch
from typing import Dict, List, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from utils.log.hybridlogging import get_hybrid_logger
import onnxruntime as ort
import os

class SemanticAnalysisEngine:
    """
    순수 의미 분석 유틸리티 - CLIP 모델을 사용하여 텍스트와 이미지 간의 의미적 유사도를 계산.
    ✅ CLIP 모델 중복 방지 및 외부 세션 공유 지원
    """
    
    def __init__(self, logger: Any, shared_clip_session: Optional[Dict] = None):
        self.logger = logger
        self.device = "cpu"
        
        # ✅ 외부 CLIP 세션 공유 지원
        if shared_clip_session:
            self._setup_shared_clip_models(shared_clip_session)
        else:
            self._setup_clip_models()

    def _setup_shared_clip_models(self, shared_session: Dict):
        """✅ 공유된 CLIP 모델 세션 사용 (중복 방지)"""
        try:
            self.clip_model = shared_session.get("clip_model")
            self.clip_preprocess = shared_session.get("clip_preprocess")
            self.onnx_session = shared_session.get("onnx_session")
            
            if self.clip_model and self.onnx_session:
                self.clip_available = True
                self.logger.info("✅ SemanticAnalysisEngine: 공유 CLIP 세션 연결 성공")
            else:
                self.clip_available = False
                self.logger.warning("공유 CLIP 세션이 불완전합니다.")
                
        except Exception as e:
            self.logger.error(f"공유 CLIP 세션 설정 실패: {e}")
            self.clip_available = False

    def _setup_clip_models(self):
        """CLIP 모델(PyTorch 텍스트용, ONNX 이미지용)을 설정합니다."""
        self.clip_available = False
        self.clip_model = None
        self.clip_preprocess = None
        self.onnx_session = None
        
        onnx_model_path = "models/clip_onnx/clip_visual.quant.onnx"

        try:
            # 텍스트 인코딩용 PyTorch 모델
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='laion2b_s34b_b79k', device=self.device
            )
            self.clip_model.eval()
            self.logger.info("✅ SemanticAnalysisEngine: PyTorch CLIP 모델(Text) 초기화 성공")

            # 이미지 인코딩용 ONNX 모델
            if os.path.exists(onnx_model_path):
                self.onnx_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
                self.logger.info("✅ SemanticAnalysisEngine: ONNX CLIP 모델(Visual) 로드 성공")
                self.clip_available = True
            else:
                self.logger.warning(f"ONNX 모델 파일을 찾을 수 없어 이미지 분석이 제한됩니다: {onnx_model_path}")
        except Exception as e:
            self.logger.error(f"CLIP 모델 설정 실패: {e}")
            self.clip_available = False

    def get_clip_session(self) -> Dict:
        """✅ CLIP 세션 공유를 위한 메서드"""
        return {
            "clip_model": self.clip_model,
            "clip_preprocess": self.clip_preprocess,
            "onnx_session": self.onnx_session,
            "clip_available": self.clip_available
        }

    async def calculate_semantic_similarity(self, texts: List[str], images: List[Dict]) -> Dict:
        """
        주어진 텍스트와 이미지 리스트 간의 코사인 유사도 행렬을 계산합니다.
        
        반환값:
        {
            "similarity_matrix": np.ndarray,  # (n_texts, n_images) 형태의 유사도 행렬
            "text_embeddings": np.ndarray,
            "image_embeddings": np.ndarray
        }
        """
        if not self.clip_available:
            self.logger.warning("CLIP 모델을 사용할 수 없어 의미 유사도를 계산할 수 없습니다.")
            return {"similarity_matrix": np.array([]), "text_embeddings": np.array([]), "image_embeddings": np.array([])}

        if not texts or not images:
            self.logger.info("텍스트 또는 이미지가 제공되지 않아 유사도 계산을 건너뜁니다.")
            return {"similarity_matrix": np.array([]), "text_embeddings": np.array([]), "image_embeddings": np.array([])}
        
        try:
            # 텍스트와 이미지 임베딩을 병렬로 생성
            text_embedding_task = self._generate_clip_text_embeddings(texts)
            image_embedding_task = self._generate_clip_image_embeddings_from_data(images)
            
            text_embeddings, image_embeddings = await asyncio.gather(
                text_embedding_task, image_embedding_task
            )

            # 임베딩 생성 실패 시 처리
            if text_embeddings.size == 0 or image_embeddings.size == 0:
                self.logger.error("텍스트 또는 이미지 임베딩 생성에 실패했습니다.")
                return {"similarity_matrix": np.array([]), "text_embeddings": text_embeddings, "image_embeddings": image_embeddings}

            # 코사인 유사도 계산
            similarity_matrix = cosine_similarity(text_embeddings, image_embeddings)

            return {
                "similarity_matrix": similarity_matrix,
                "text_embeddings": text_embeddings,
                "image_embeddings": image_embeddings
            }
        except Exception as e:
            self.logger.error(f"의미 유사도 계산 중 오류 발생: {e}")
            return {"similarity_matrix": np.array([]), "text_embeddings": np.array([]), "image_embeddings": np.array([])}

    async def _generate_clip_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """텍스트 리스트에 대한 CLIP 텍스트 임베딩 생성 (PyTorch 사용)"""
        if self.clip_model is None or not texts:
            return np.zeros((len(texts), 512), dtype=np.float32) if texts else np.array([])
        try:
            with torch.no_grad():
                text_tokens = open_clip.tokenize(texts).to(self.device)
                text_features = self.clip_model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                return text_features.cpu().numpy().astype(np.float32)
        except Exception as e:
            self.logger.error(f"CLIP 텍스트 임베딩 생성 오류: {e}")
            return np.zeros((len(texts), 512), dtype=np.float32)

    async def _generate_clip_image_embeddings_from_data(self, images: List[Dict]) -> np.ndarray:
        """주어진 이미지 데이터 리스트에서 CLIP 이미지 임베딩을 생성 (ONNX 사용)"""
        if self.onnx_session is None or not images:
            return np.zeros((len(images), 512), dtype=np.float32) if images else np.array([])
        
        async def fetch_image(session, idx, url):
            try:
                async with session.get(url, timeout=10) as response:
                    response.raise_for_status()
                    content = await response.read()
                    return idx, content
            except Exception:
                return idx, None

        pil_images_to_process, indices_for_pil = [], []
        image_embeddings = [None] * len(images)

        download_info = [(i, img.get("image_url")) for i, img in enumerate(images) if img.get("image_url")]

        async with aiohttp.ClientSession() as session:
            tasks = [fetch_image(session, idx, url) for idx, url in download_info]
            results = await asyncio.gather(*tasks)
            for idx, content in results:
                if content:
                    pil_images_to_process.append(Image.open(BytesIO(content)).convert("RGB"))
                    indices_for_pil.append(idx)
        
        if pil_images_to_process:
            image_tensors = torch.stack([self.clip_preprocess(img) for img in pil_images_to_process])
            onnx_inputs = {self.onnx_session.get_inputs()[0].name: image_tensors.cpu().numpy()}
            batch_embeddings = self.onnx_session.run(None, onnx_inputs)[0]
            norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
            normalized_embeddings = batch_embeddings / (norms + 1e-12)
            for i, original_idx in enumerate(indices_for_pil):
                image_embeddings[original_idx] = normalized_embeddings[i]
        
        # None으로 남은 임베딩을 0 벡터로 채움
        for i in range(len(images)):
            if image_embeddings[i] is None:
                image_embeddings[i] = np.zeros(512, dtype=np.float32)

        return np.array(image_embeddings).astype(np.float32)
