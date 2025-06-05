from uuid import uuid4


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
    except Exception as e:
        print(f"❌ Cosmos DB 저장 실패: {e}")
