from uuid import uuid4
from azure.cosmos.exceptions import CosmosResourceNotFoundError
from typing import List, Dict
from datetime import datetime

import uuid


def save_to_cosmos(container, data, partition_key_field):
    """
    container: Cosmos DB 컨테이너 객체
    data: dict (저장할 데이터)
    partition_key_field: str (파티션 키로 사용할 필드명, 예: '이미지 - user_id' 또는 '매거진 - mag_id' , '로깅-session_id')
    """
    try:
        # id 필드가 없으면 생성
        if 'id' not in data:
            data['id'] = str(uuid4())
        # 파티션 키 필드가 없으면 생성 (id와 동일하게 할 수도 있음)
        if partition_key_field and partition_key_field not in data:
            data[partition_key_field] = data['id']

        # ✅ partition_key 파라미터 제거
        container.upsert_item(data)
        print(
            f"✅ Cosmos DB 저장 성공: {data['id']} (partition: {partition_key_field}={data[partition_key_field]})")
        return data['id']
    except Exception as e:
        print(f"❌ Cosmos DB 저장 실패: {e}")
        return None


def get_from_cosmos(container, item_id, partition_key=None):
    """
    Cosmos DB에서 특정 아이템을 조회합니다.
    
    container: Cosmos DB 컨테이너 객체
    item_id: 조회할 아이템의 id
    partition_key: 파티션 키 값 (파티션 키 필드의 값, 예: 특정 user_id 또는 mag_id)
    """
    try:
        if partition_key:
            # 파티션 키가 주어진 경우 - 효율적인 조회
            item = container.read_item(item=item_id, partition_key=partition_key)
        else:
            # 파티션 키를 모르는 경우 - 쿼리로 조회 (비효율적)
            query = f"SELECT * FROM c WHERE c.id = '{item_id}'"
            items = list(container.query_items(query=query, enable_cross_partition_query=True))
            if not items:
                return None
            item = items[0]
        
        return item
    except CosmosResourceNotFoundError:
        return None
    except Exception as e:
        print(f"❌ Cosmos DB 조회 중 예상치 못한 오류 발생: {e}")
        return None


def save_jsx_components(container, magazine_id: str, components: List[Dict], 
                       order_matters: bool = True, session_id: str = None):
    """JSX 컴포넌트를 CosmosDB에 저장 (세션 ID 포함)"""
    saved_ids = []
    
    # session_id가 제공되지 않으면 새로 생성
    if not session_id:
        session_id = str(uuid.uuid4())
    
    for idx, component in enumerate(components):
        # 고유 ID 생성
        component_id = f"{magazine_id}_session_{session_id}_component_{idx:03d}"
        
        # CosmosDB 문서 구성
        document = {
            "id": component_id,
            "magazine_id": magazine_id,
            "session_id": session_id,  # ✅ 세션 ID 추가
            "order_index": idx if order_matters else 0,
            "title": component.get("title", f"섹션 {idx + 1}"),
            "jsx_code": component.get("jsx_code", ""),
            "metadata": component.get("metadata", {}),
            "created_at": component.get("created_at", datetime.now().isoformat()),
            "_partition_key": magazine_id  # 파티션 키
        }
        
        try:
            # CosmosDB에 저장
            created_item = container.create_item(document)
            saved_ids.append(created_item["id"])
        except Exception as e:

            continue
    
    return saved_ids


def update_agent_logs_in_cosmos(container, session_id, agent_name, output_data):
    """
    에이전트 로그를 Cosmos DB의 로깅 컨테이너에 업데이트합니다.
    session_id를 기준으로 하나의 문서에 지속적으로 추가/업데이트합니다.
    
    container: Cosmos DB 로깅 컨테이너 객체
    session_id: 세션 ID (파티션 키로 사용)
    agent_name: 에이전트 이름
    output_data: 에이전트 출력 데이터
    
    반환값: 로그 문서 ID
    """
    try:
        # 1. 기존 로그 문서 조회
        document_id = f"agent_logs_{session_id}"
        existing_doc = get_from_cosmos(container, document_id, partition_key=session_id)
        
        # 2. 문서가 없으면 새로 생성
        if not existing_doc:
            new_doc = {
                "id": document_id,
                "session_id": session_id,
                "document_type": "agent_logs",
                "created_at": __import__('datetime').datetime.now().isoformat(),
                "updated_at": __import__('datetime').datetime.now().isoformat(),
                "agent_outputs": {},
                "metadata": {
                    "total_outputs": 0,
                    "agents": []
                }
            }
            existing_doc = new_doc
        
        # 3. 에이전트 출력 업데이트
        if agent_name not in existing_doc["agent_outputs"]:
            existing_doc["agent_outputs"][agent_name] = []
            if agent_name not in existing_doc["metadata"]["agents"]:
                existing_doc["metadata"]["agents"].append(agent_name)
        
        # 출력 데이터에 타임스탬프 추가
        if "timestamp" not in output_data:
            output_data["timestamp"] = __import__('datetime').datetime.now().isoformat()
        
        # 출력 ID 추가
        if "output_id" not in output_data:
            output_data["output_id"] = f"{agent_name}_{len(existing_doc['agent_outputs'][agent_name])}_{int(__import__('time').time())}"
        
        # 출력 추가
        existing_doc["agent_outputs"][agent_name].append(output_data)
        existing_doc["metadata"]["total_outputs"] += 1
        existing_doc["updated_at"] = __import__('datetime').datetime.now().isoformat()
        
        # 4. 문서 업데이트
        container.upsert_item(existing_doc)
        print(f"✅ 에이전트 로그 업데이트 성공: {document_id} (agent: {agent_name})")
        return document_id
        
    except Exception as e:
        print(f"❌ 에이전트 로그 업데이트 실패: {e}")
        return None


def get_agent_logs_from_cosmos(container, session_id, agent_name=None):
    """
    Cosmos DB에서 에이전트 로그를 조회합니다.
    
    container: Cosmos DB 로깅 컨테이너 객체
    session_id: 세션 ID
    agent_name: 특정 에이전트의 로그만 조회할 경우 지정
    
    반환값: 전체 로그 문서 또는 특정 에이전트의 로그 목록
    """
    try:
        document_id = f"agent_logs_{session_id}"
        doc = get_from_cosmos(container, document_id, partition_key=session_id)
        
        if not doc:
            return None
            
        if agent_name:
            # 특정 에이전트의 로그만 반환
            return doc.get("agent_outputs", {}).get(agent_name, [])
        else:
            # 전체 로그 문서 반환
            return doc
            
    except Exception as e:
        print(f"❌ 에이전트 로그 조회 실패: {e}")
        return None
