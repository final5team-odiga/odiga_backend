# Odiga Backend 프로젝트

FastAPI 기반의 백엔드 API와 통합 대시보드 UI 시스템입니다.

## 프로젝트 구조

```
project_root/
├── backend/              # 백엔드 API 코드
│   └── app/
│       ├── api/          # API 라우터
│       ├── crud/         # CRUD 기능
│       ├── db/           # 데이터베이스 관련
│       ├── models/       # 데이터 모델
│       ├── utils/        # 유틸리티 함수
│       └── main.py       # 백엔드 애플리케이션
├── static/               # 정적 파일
│   ├── css/              # CSS 스타일시트
│   └── js/               # JavaScript 파일
├── templates/            # HTML 템플릿
│   └── index.html        # 통합 대시보드 UI
├── main.py               # 메인 애플리케이션 진입점
└── README.md             # 프로젝트 설명서
```

## 시작하기

### 필수 조건

- Python 3.8 이상
- PostgreSQL 데이터베이스

### 환경 설정

1. 환경 변수 파일 (.env) 생성:

```
DATABASE_URL=postgresql+asyncpg://username:password@localhost/dbname
```

2. 필요한 패키지 설치:

```bash
pip install -r backend/app/requirements.txt
```

### 실행 방법

1. 애플리케이션 실행:

```bash
python main.py
```

2. 브라우저에서 접속:
   - 대시보드 UI: http://localhost:8000/
   - API 문서: http://localhost:8000/docs 또는 http://localhost:8000/redoc

## API 엔드포인트

### 인증 API

| 메서드 | 경로                | 설명                | 인증 필요 |
| ------ | ------------------- | ------------------- | --------- |
| POST   | /auth/signup/       | 사용자 등록         | 아니오    |
| POST   | /auth/login/        | 로그인              | 아니오    |
| POST   | /auth/logout/       | 로그아웃            | 예        |
| GET    | /auth/check_userid/ | 사용자 ID 중복 확인 | 아니오    |

### 게시글 API

| 메서드 | 경로                       | 설명             | 인증 필요 |
| ------ | -------------------------- | ---------------- | --------- |
| GET    | /articles/                 | 게시글 목록 조회 | 아니오    |
| POST   | /articles/                 | 새 게시글 작성   | 예        |
| GET    | /articles/{articleID}      | 특정 게시글 조회 | 아니오    |
| PUT    | /articles/{articleID}      | 게시글 수정      | 예        |
| DELETE | /articles/{articleID}      | 게시글 삭제      | 예        |
| POST   | /articles/{articleID}/like | 게시글 좋아요    | 예        |

### 댓글 API

| 메서드 | 경로                                       | 설명           | 인증 필요 |
| ------ | ------------------------------------------ | -------------- | --------- |
| GET    | /articles/{articleID}/comments             | 댓글 목록 조회 | 아니오    |
| POST   | /articles/{articleID}/comments             | 새 댓글 작성   | 예        |
| PUT    | /articles/{articleID}/comments/{commentID} | 댓글 수정      | 예        |
| DELETE | /articles/{articleID}/comments/{commentID} | 댓글 삭제      | 예        |

### 프로필 API

| 메서드 | 경로                            | 설명                 | 인증 필요 |
| ------ | ------------------------------- | -------------------- | --------- |
| GET    | /profiles/{userID}              | 프로필 조회          | 아니오    |
| PUT    | /profiles/{userID}              | 프로필 수정          | 예        |
| DELETE | /profiles/{userID}              | 회원 탈퇴            | 예        |
| POST   | /profiles/{userID}/upload-image | 프로필 이미지 업로드 | 예        |

### 음성 API

| 메서드 | 경로               | 설명                   | 인증 필요 |
| ------ | ------------------ | ---------------------- | --------- |
| POST   | /speech/tts        | 텍스트를 음성으로 변환 | 예        |
| POST   | /speech/transcribe | 음성을 텍스트로 변환   | 예        |

### 매거진 API

| 메서드 | 경로               | 설명               | 인증 필요 |
| ------ | ------------------ | ------------------ | --------- |
| POST   | /magazine/generate | 매거진 생성        | 예        |
| GET    | /magazine/{userID} | 생성된 매거진 조회 | 예        |

## 통합 대시보드 사용법

통합 대시보드 UI는 모든 API 엔드포인트를 테스트하고 모니터링할 수 있는 기능을 제공합니다:

1. **API 테스트**

   - 각 API 엔드포인트에 대한 테스트 폼 제공
   - 요청/응답 결과 실시간 확인
   - 성공/실패 상태 모니터링

2. **데이터 시각화**

   - API 호출 통계 차트
   - 응답 시간 추이 그래프
   - 성공률 모니터링

3. **로그 모니터링**

   - 최근 API 호출 로그 확인
   - 에러 발생 시 즉시 알림

4. **배치 테스트**
   - 여러 API를 순차적으로 테스트
   - 통합 시나리오 테스트 기능

## 보안 고려사항

- 모든 비밀번호는 bcrypt로 해싱됩니다.
- 인증이 필요한 API는 세션 쿠키를 통해 보호됩니다.
- XSS 및 CSRF 공격 방지를 위한 보안 조치가 구현되어 있습니다.

## 기여 방법

1. 이 저장소를 포크합니다.
2. 새 브랜치를 생성합니다: `git checkout -b feature/your-feature-name`
3. 변경사항을 커밋합니다: `git commit -m 'Add some feature'`
4. 브랜치를 푸시합니다: `git push origin feature/your-feature-name`
5. Pull Request를 제출합니다.

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.
