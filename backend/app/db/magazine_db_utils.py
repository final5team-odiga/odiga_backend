from typing import Dict, List, Optional
from .cosmos_connection import magazine_container, image_container
from uuid import uuid4
from datetime import datetime

class MagazineDBUtils:
    @staticmethod
    async def get_magazine_by_id(magazine_id: str) -> Optional[Dict]:
        """매거진 ID로 매거진 데이터 조회"""
        try:
            query = f"SELECT * FROM c WHERE c.id = '{magazine_id}'"
            items = list(magazine_container.query_items(query=query, enable_cross_partition_query=True))
            if items:
                return items[0]
            else:
                print(f"⚠️ 매거진 ID {magazine_id}에 해당하는 문서를 찾을 수 없습니다.")
                return None
        except Exception as e:
            print(f"❌ 매거진 조회 실패: {e}")
            return None

    @staticmethod
    async def get_images_by_magazine_id(magazine_id: str) -> List[Dict]:
        """매거진 ID로 관련 이미지 분석 결과 조회"""
        try:
            # 변경: 단일 문서에서 이미지 분석 결과 배열 조회
            query = f"SELECT * FROM c WHERE c.magazine_id = '{magazine_id}'"
            items = list(image_container.query_items(query=query, enable_cross_partition_query=True))
            
            # 단일 통합 문서가 있으면 그 안의 이미지 분석 배열 반환
            for item in items:
                if "image_analyses" in item:
                    print(f"✅ 통합 이미지 분석 문서에서 {len(item['image_analyses'])}개의 이미지 분석 결과 조회 완료")
                    return item["image_analyses"]
            
            # 이전 방식의 개별 이미지 분석 결과가 있는 경우 (하위 호환성)
            if items:
                print(f"⚠️ 개별 이미지 분석 문서 {len(items)}개 조회됨 (이전 방식)")
                return items
            
            print(f"⚠️ 매거진 ID {magazine_id}에 대한 이미지 분석 결과가 없습니다. 빈 배열 반환.")
            return []
        except Exception as e:
            print(f"이미지 조회 실패: {e}")
            print(f"⚠️ 이미지 조회 실패 시 빈 배열 반환")
            return []

    @staticmethod
    async def save_magazine_content(content: Dict) -> Optional[str]:
        """매거진 콘텐츠 저장"""
        try:
            # id 필드가 없으면 자동으로 생성
            if 'id' not in content:
                content['id'] = str(uuid4())
            
            result = magazine_container.create_item(body=content)
            print(f"✅ 매거진 콘텐츠 저장 성공: {result['id']}")
            return result['id']
        except Exception as e:
            print(f"❌ 매거진 저장 실패: {e}")
            return None

    @staticmethod
    async def save_image_analysis(analysis: Dict, magazine_id: str) -> Optional[str]:
        """이미지 분석 결과를 이미지 컨테이너에 저장 (통합 문서 방식 사용)"""
        try:
            # 통합 문서 형식으로 변환하여 저장
            combined_analysis = {
                "id": str(uuid4()),
                "magazine_id": magazine_id,
                "created_at": str(datetime.now()),
                "analysis_count": 1,
                "image_analyses": [analysis]
            }
            
            return await MagazineDBUtils.save_combined_image_analysis(combined_analysis)
        except Exception as e:
            print(f"이미지 분석 저장 실패: {e}")
            return None

    @staticmethod
    async def update_magazine_content(magazine_id: str, content: Dict) -> bool:
        """매거진 콘텐츠 업데이트"""
        try:
            # 기존 매거진 데이터 조회
            magazine_data = await MagazineDBUtils.get_magazine_by_id(magazine_id)
            if not magazine_data:
                print(f"⚠️ 매거진 ID {magazine_id}에 해당하는 문서를 찾을 수 없습니다. 새로 생성합니다.")
                # 문서가 없으면 새로 생성
                content['id'] = magazine_id
                await MagazineDBUtils.save_magazine_content(content)
                return True
            
            # 새로운 콘텐츠로 기존 데이터 업데이트
            for key, value in content.items():
                magazine_data[key] = value
                
            magazine_data['id'] = magazine_id
            magazine_container.replace_item(item=magazine_id, body=magazine_data)
            print(f"✅ 매거진 ID {magazine_id} 업데이트 완료")
            return True
        except Exception as e:
            print(f"❌ 매거진 업데이트 실패: {e}")
            return False

    @staticmethod
    async def save_combined_image_analysis(combined_analysis: Dict) -> Optional[str]:
        """이미지 분석 결과를 하나의 통합 문서로 이미지 컨테이너에 저장"""
        try:
            # 이미지 컨테이너에 저장
            result = image_container.create_item(body=combined_analysis)
            print(f"✅ 통합 이미지 분석 결과 ({combined_analysis['analysis_count']}개) 이미지 컨테이너에 저장 완료: {result['id']}")
            return result['id']
        except Exception as e:
            print(f"통합 이미지 분석 저장 실패: {e}")
            return None 