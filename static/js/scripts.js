/**
 * Odiga Backend Dashboard
 * 프론트엔드 통합 인터페이스를 위한 JavaScript
 */

// 전역 변수
let apiCallStats = {};
let apiResponseTimes = [];
let darkMode = localStorage.getItem("darkMode") === "true";

// DOM이 로드된 후 초기화
document.addEventListener("DOMContentLoaded", () => {
  // 다크 모드 초기화
  if (darkMode) {
    document.body.classList.add("dark-mode");
    document.getElementById("theme-toggle-btn").textContent = "☀️";
  }

  // 사이드바 네비게이션 기능
  initNavigation();

  // API 테스트 폼 이벤트 리스너
  initApiTestForms();

  // 다크 모드 토글 버튼
  document
    .getElementById("theme-toggle-btn")
    .addEventListener("click", toggleDarkMode);

  // 대시보드 차트 초기화
  initDashboardCharts();

  // 로그 테이블 초기화
  initLogsTable();

  // API 검색 기능
  document
    .getElementById("api-search")
    .addEventListener("input", searchApiEndpoints);
});

/**
 * 사이드바 네비게이션 초기화
 */
function initNavigation() {
  const navLinks = document.querySelectorAll(".nav-link");
  const sections = document.querySelectorAll(".content-section");

  navLinks.forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();

      // 활성 네비게이션 링크 설정
      navLinks.forEach((l) => l.classList.remove("active"));
      link.classList.add("active");

      // 활성 섹션 표시
      const targetSection = link.getAttribute("data-section");
      sections.forEach((section) => {
        section.classList.remove("active");
        if (section.id === targetSection) {
          section.classList.add("active");
        }
      });
    });
  });
}

/**
 * API 테스트 폼 초기화
 */
function initApiTestForms() {
  const sendButtons = document.querySelectorAll(".send-btn");

  sendButtons.forEach((button) => {
    button.addEventListener("click", async (e) => {
      const endpoint = button.getAttribute("data-endpoint");
      const responseBox = document.getElementById(`${endpoint}-response`);

      // 요청 준비
      let url, method, formData;

      // 엔드포인트별 설정
      switch (endpoint) {
        case "auth-signup":
          url = "/auth/signup/";
          method = "POST";
          formData = new FormData();
          formData.append(
            "userID",
            document.getElementById("auth-signup-userID").value
          );
          formData.append(
            "userName",
            document.getElementById("auth-signup-userName").value
          );
          formData.append(
            "password",
            document.getElementById("auth-signup-password").value
          );
          formData.append(
            "userEmail",
            document.getElementById("auth-signup-userEmail").value
          );
          formData.append(
            "userCountry",
            document.getElementById("auth-signup-userCountry").value
          );
          formData.append(
            "userLanguage",
            document.getElementById("auth-signup-userLanguage").value
          );
          break;

        case "auth-login":
          url = "/auth/login/";
          method = "POST";
          formData = new FormData();
          formData.append(
            "userID",
            document.getElementById("auth-login-userID").value
          );
          formData.append(
            "password",
            document.getElementById("auth-login-password").value
          );
          break;

        case "auth-logout":
          url = "/auth/logout/";
          method = "POST";
          formData = new FormData();
          break;

        case "auth-check":
          url = `/auth/check_userid/?userID=${
            document.getElementById("auth-check-userID").value
          }`;
          method = "GET";
          break;

        case "articles-list":
          url = "/articles/";
          method = "GET";
          break;

        case "articles-create":
          url = "/articles/";
          method = "POST";
          formData = new FormData();
          formData.append(
            "articleTitle",
            document.getElementById("articles-create-title").value
          );
          formData.append(
            "content",
            document.getElementById("articles-create-content").value
          );
          formData.append(
            "travelCountry",
            document.getElementById("articles-create-country").value
          );
          formData.append(
            "travelCity",
            document.getElementById("articles-create-city").value
          );
          break;

        case "articles-get":
          const articleId = document.getElementById("articles-get-id").value;
          url = `/articles/${articleId}`;
          method = "GET";
          break;

        case "articles-update":
          const updateId = document.getElementById("articles-update-id").value;
          url = `/articles/${updateId}`;
          method = "PUT";
          formData = new FormData();
          formData.append(
            "articleTitle",
            document.getElementById("articles-update-title").value
          );
          formData.append(
            "content",
            document.getElementById("articles-update-content").value
          );
          formData.append(
            "travelCountry",
            document.getElementById("articles-update-country").value
          );
          formData.append(
            "travelCity",
            document.getElementById("articles-update-city").value
          );
          break;

        case "articles-delete":
          const deleteId = document.getElementById("articles-delete-id").value;
          url = `/articles/${deleteId}`;
          method = "DELETE";
          break;

        // 매거진 API 엔드포인트 추가
        case "magazine-generate":
          url = "/magazine/generate/";
          method = "POST";
          formData = new FormData();
          formData.append(
            "magazine_id",
            document.getElementById("magazine-generate-id").value
          );
          formData.append(
            "user_input",
            document.getElementById("magazine-generate-input").value
          );

          const imageFolder = document.getElementById(
            "magazine-generate-folder"
          ).value;
          if (imageFolder) {
            formData.append("image_folder", imageFolder);
          }

          formData.append(
            "generate_pdf",
            document.getElementById("magazine-generate-pdf").value
          );
          break;

        case "magazine-generate-async":
          url = "/magazine/generate-async/";
          method = "POST";
          formData = new FormData();
          formData.append(
            "magazine_id",
            document.getElementById("magazine-generate-async-id").value
          );
          formData.append(
            "user_input",
            document.getElementById("magazine-generate-async-input").value
          );

          const asyncImageFolder = document.getElementById(
            "magazine-generate-async-folder"
          ).value;
          if (asyncImageFolder) {
            formData.append("image_folder", asyncImageFolder);
          }

          formData.append(
            "generate_pdf",
            document.getElementById("magazine-generate-async-pdf").value
          );
          break;

        case "magazine-status":
          const magazineId =
            document.getElementById("magazine-status-id").value;
          url = `/magazine/status/${magazineId}`;
          method = "GET";
          break;

        // 분석 API 엔드포인트 추가
        case "analytics-country-counts":
          url = "/analytics/country-counts/";
          method = "GET";
          break;

        // 스토리지 API 엔드포인트 추가
        case "storage-images-list":
          const listMagazineId = document.getElementById(
            "storage-images-magazineID"
          ).value;
          url = `/storage/images/?magazine_id=${listMagazineId}`;
          method = "GET";
          break;

        case "storage-images-upload":
          url = "/storage/images/upload/";
          method = "POST";
          formData = new FormData();
          formData.append(
            "magazine_id",
            document.getElementById("storage-images-upload-magazineID").value
          );

          const fileInput = document.getElementById(
            "storage-images-upload-files"
          );
          if (fileInput.files.length > 0) {
            for (let i = 0; i < fileInput.files.length; i++) {
              formData.append("files", fileInput.files[i]);
            }
          } else {
            showToast("업로드할 파일을 선택해주세요.", "warning");
            responseBox.innerHTML = `<pre class="error">Error: 업로드할 파일이 선택되지 않았습니다.</pre>`;
            return;
          }
          break;

        case "storage-images-delete":
          url = "/storage/images/delete/";
          method = "DELETE";
          formData = new FormData();
          formData.append(
            "magazine_id",
            document.getElementById("storage-images-delete-magazineID").value
          );
          formData.append(
            "filename",
            document.getElementById("storage-images-delete-filename").value
          );
          break;

        // 음성 API 엔드포인트 추가
        case "speech-transcribe":
          url = "/speech/transcribe/";
          method = "POST";
          formData = new FormData();

          const audioFile = document.getElementById("speech-transcribe-file")
            .files[0];
          if (!audioFile) {
            showToast("변환할 오디오 파일을 선택해주세요.", "warning");
            responseBox.innerHTML = `<pre class="error">Error: 오디오 파일이 선택되지 않았습니다.</pre>`;
            return;
          }

          formData.append("audio_file", audioFile);
          break;

        case "speech-tts":
          url = "/speech/tts/";
          method = "POST";
          formData = new FormData();

          const ttsText = document
            .getElementById("speech-tts-text")
            .value.trim();
          if (!ttsText) {
            showToast("변환할 텍스트를 입력해주세요.", "warning");
            responseBox.innerHTML = `<pre class="error">Error: 텍스트가 입력되지 않았습니다.</pre>`;
            return;
          }

          formData.append("text_input", ttsText);
          break;

        // 여기에 다른 API 엔드포인트 추가

        default:
          showToast("알 수 없는 엔드포인트입니다.", "error");
          return;
      }

      // API 호출 실행
      try {
        // 로딩 표시
        responseBox.innerHTML = '<div class="spinner"></div> 요청 처리 중...';

        // 시작 시간 기록
        const startTime = performance.now();

        // API 요청
        let response;
        const options = {
          method: method,
          credentials: "include",
        };

        if (formData && method !== "GET") {
          options.body = formData;
        }

        if (method === "GET") {
          response = await fetch(url);
        } else {
          response = await fetch(url, options);
        }

        // 종료 시간 및 응답 시간 계산
        const endTime = performance.now();
        const responseTime = endTime - startTime;

        // 응답 데이터 처리
        const contentType = response.headers.get("content-type");
        let data;

        if (contentType && contentType.includes("application/json")) {
          data = await response.json();
        } else {
          data = await response.text();
        }

        // 응답 표시
        responseBox.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;

        // 특별한 처리: 매거진 생성 비동기일 경우 자동 폴링 시작
        if (endpoint === "magazine-generate-async" && response.ok) {
          if (data.magazine_id) {
            showToast(
              "매거진 생성이 시작되었습니다. 상태를 자동으로 확인합니다.",
              "info"
            );
            // 5초마다 상태 확인 (최대 5분)
            startMagazineStatusPolling(data.magazine_id, 60);
          }
        }

        // 특별한 처리: 국가별 게시글 수 차트 생성
        if (
          endpoint === "analytics-country-counts" &&
          response.ok &&
          data.success
        ) {
          renderCountryChart(data.data);
        }

        // 특별한 처리: 이미지 목록 결과 표시
        if (endpoint === "storage-images-list" && response.ok && data.success) {
          renderImageGallery(data.images, "storage-images-list-response");
        }

        // 특별한 처리: TTS 결과 오디오 재생
        if (endpoint === "speech-tts" && response.ok && data.success) {
          renderAudioPlayer(data.audio_data_uri, "speech-tts-audio-container");
        }

        // 성공 시 토스트 표시
        if (response.ok) {
          showToast(`API 호출 성공 (${Math.round(responseTime)}ms)`, "success");
        } else {
          showToast(
            `API 호출 실패: ${response.status} ${response.statusText}`,
            "error"
          );
        }

        // 응답 통계 업데이트
        updateApiStats(url, responseTime, response.ok);
      } catch (error) {
        responseBox.innerHTML = `<pre class="error">Error: ${error.message}</pre>`;
        showToast(`API 호출 에러: ${error.message}`, "error");
      }
    });
  });
}

/**
 * 다크 모드 토글
 */
function toggleDarkMode() {
  darkMode = !darkMode;
  document.body.classList.toggle("dark-mode");

  const themeButton = document.getElementById("theme-toggle-btn");
  themeButton.textContent = darkMode ? "☀️" : "🌙";

  localStorage.setItem("darkMode", darkMode);

  // 차트 업데이트 (다크 모드에 맞게)
  updateChartsTheme();
}

/**
 * 차트 테마 업데이트
 */
function updateChartsTheme() {
  // 존재하는 차트 업데이트
  if (window.apiCallsChart) {
    window.apiCallsChart.update();
  }
  if (window.responseTimeChart) {
    window.responseTimeChart.update();
  }
}

/**
 * 대시보드 차트 초기화
 */
function initDashboardCharts() {
  // API 호출 차트
  const apiCallsCtx = document.getElementById("apiCallsChart").getContext("2d");
  window.apiCallsChart = new Chart(apiCallsCtx, {
    type: "bar",
    data: {
      labels: ["GET", "POST", "PUT", "DELETE"],
      datasets: [
        {
          label: "API 호출 수",
          data: [12, 8, 5, 2],
          backgroundColor: [
            "rgba(54, 162, 235, 0.6)",
            "rgba(75, 192, 192, 0.6)",
            "rgba(255, 206, 86, 0.6)",
            "rgba(255, 99, 132, 0.6)",
          ],
          borderColor: [
            "rgba(54, 162, 235, 1)",
            "rgba(75, 192, 192, 1)",
            "rgba(255, 206, 86, 1)",
            "rgba(255, 99, 132, 1)",
          ],
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          grid: {
            color: darkMode ? "rgba(255, 255, 255, 0.1)" : "rgba(0, 0, 0, 0.1)",
          },
          ticks: {
            color: darkMode ? "#9ca3af" : "#6b7280",
          },
        },
        x: {
          grid: {
            display: false,
          },
          ticks: {
            color: darkMode ? "#9ca3af" : "#6b7280",
          },
        },
      },
      plugins: {
        legend: {
          labels: {
            color: darkMode ? "#f3f4f6" : "#1f2937",
          },
        },
      },
    },
  });

  // 응답 시간 차트
  const responseTimeCtx = document
    .getElementById("responseTimeChart")
    .getContext("2d");
  window.responseTimeChart = new Chart(responseTimeCtx, {
    type: "line",
    data: {
      labels: [
        "1분 전",
        "50초 전",
        "40초 전",
        "30초 전",
        "20초 전",
        "10초 전",
        "현재",
      ],
      datasets: [
        {
          label: "평균 응답 시간 (ms)",
          data: [320, 280, 300, 250, 210, 240, 230],
          backgroundColor: "rgba(62, 99, 221, 0.2)",
          borderColor: "rgba(62, 99, 221, 1)",
          borderWidth: 2,
          tension: 0.3,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          grid: {
            color: darkMode ? "rgba(255, 255, 255, 0.1)" : "rgba(0, 0, 0, 0.1)",
          },
          ticks: {
            color: darkMode ? "#9ca3af" : "#6b7280",
          },
        },
        x: {
          grid: {
            display: false,
          },
          ticks: {
            color: darkMode ? "#9ca3af" : "#6b7280",
          },
        },
      },
      plugins: {
        legend: {
          labels: {
            color: darkMode ? "#f3f4f6" : "#1f2937",
          },
        },
      },
    },
  });
}

/**
 * 로그 테이블 초기화
 */
function initLogsTable() {
  const logsBody = document.getElementById("logs-body");

  // 샘플 로그 데이터
  const sampleLogs = [
    {
      time: "13:45:22",
      endpoint: "/auth/login/",
      method: "POST",
      status: 200,
      responseTime: 128,
    },
    {
      time: "13:44:15",
      endpoint: "/articles/",
      method: "GET",
      status: 200,
      responseTime: 95,
    },
    {
      time: "13:42:56",
      endpoint: "/articles/123",
      method: "GET",
      status: 200,
      responseTime: 108,
    },
    {
      time: "13:40:31",
      endpoint: "/articles/",
      method: "POST",
      status: 201,
      responseTime: 245,
    },
    {
      time: "13:38:12",
      endpoint: "/profiles/user123",
      method: "GET",
      status: 200,
      responseTime: 87,
    },
    {
      time: "13:35:45",
      endpoint: "/auth/logout/",
      method: "POST",
      status: 200,
      responseTime: 65,
    },
    {
      time: "13:32:18",
      endpoint: "/speech/tts",
      method: "POST",
      status: 200,
      responseTime: 1250,
    },
    {
      time: "13:30:02",
      endpoint: "/articles/456",
      method: "PUT",
      status: 200,
      responseTime: 185,
    },
    {
      time: "13:25:47",
      endpoint: "/articles/789",
      method: "DELETE",
      status: 204,
      responseTime: 125,
    },
    {
      time: "13:20:33",
      endpoint: "/auth/signup/",
      method: "POST",
      status: 201,
      responseTime: 215,
    },
  ];

  // 로그 행 생성
  sampleLogs.forEach((log) => {
    const row = document.createElement("tr");

    // 상태에 따른 색상 스타일 지정
    let statusClass = "text-success";
    if (log.status >= 400) statusClass = "text-danger";
    else if (log.status >= 300) statusClass = "text-warning";

    // 응답 시간에 따른 색상 스타일 지정
    let timeClass = "text-success";
    if (log.responseTime > 1000) timeClass = "text-danger";
    else if (log.responseTime > 300) timeClass = "text-warning";

    // 행 내용 설정
    row.innerHTML = `
      <td>${log.time}</td>
      <td>${log.endpoint}</td>
      <td><span class="method ${log.method.toLowerCase()}">${
      log.method
    }</span></td>
      <td class="${statusClass}">${log.status}</td>
      <td class="${timeClass}">${log.responseTime}ms</td>
    `;

    logsBody.appendChild(row);
  });
}

/**
 * API 통계 업데이트
 * @param {string} url - API URL
 * @param {number} responseTime - 응답 시간 (ms)
 * @param {boolean} success - 성공 여부
 */
function updateApiStats(url, responseTime, success) {
  if (!apiCallStats[url]) {
    apiCallStats[url] = {
      count: 0,
      totalTime: 0,
      successCount: 0,
      minTime: Infinity,
      maxTime: 0,
    };
  }

  const stats = apiCallStats[url];
  stats.count += 1;
  stats.totalTime += responseTime;
  if (success) stats.successCount += 1;
  stats.minTime = Math.min(stats.minTime, responseTime);
  stats.maxTime = Math.max(stats.maxTime, responseTime);

  // 응답 시간 배열에 추가 (최대 100개까지 저장)
  apiResponseTimes.push({
    url,
    time: responseTime,
    timestamp: new Date(),
    success,
  });

  if (apiResponseTimes.length > 100) {
    apiResponseTimes.shift();
  }

  // 응답 시간 차트 업데이트
  updateResponseTimeChart();
}

/**
 * 응답 시간 차트 업데이트
 */
function updateResponseTimeChart() {
  if (apiResponseTimes.length === 0 || !window.responseTimeChart) return;

  // 최근 7개의 데이터만 사용
  const recentTimes = apiResponseTimes.slice(-7);

  // 레이블 및 데이터 업데이트
  const labels = recentTimes.map((item) => {
    const time = item.timestamp;
    return `${time.getHours()}:${time.getMinutes()}:${time.getSeconds()}`;
  });

  const data = recentTimes.map((item) => item.time);

  window.responseTimeChart.data.labels = labels;
  window.responseTimeChart.data.datasets[0].data = data;
  window.responseTimeChart.update();
}

/**
 * API 엔드포인트 검색
 */
function searchApiEndpoints() {
  const searchTerm = document.getElementById("api-search").value.toLowerCase();
  const endpointCards = document.querySelectorAll(".endpoint-card");

  endpointCards.forEach((card) => {
    const path = card.querySelector(".path").textContent.toLowerCase();
    const method = card.querySelector(".method").textContent.toLowerCase();
    const description =
      card.querySelector(".description")?.textContent.toLowerCase() || "";

    if (
      path.includes(searchTerm) ||
      method.includes(searchTerm) ||
      description.includes(searchTerm)
    ) {
      card.style.display = "block";
    } else {
      card.style.display = "none";
    }
  });
}

/**
 * 토스트 알림 표시
 * @param {string} message - 표시할 메시지
 * @param {string} type - 알림 유형 (success, error, warning, info)
 */
function showToast(message, type = "success") {
  const toast = document.getElementById("toast");
  const toastMessage = document.getElementById("toast-message");

  // 토스트 메시지 설정
  toastMessage.textContent = message;

  // 토스트 유형 설정
  toast.className = "toast";
  toast.classList.add(type);

  // 토스트 표시
  toast.classList.add("show");

  // 자동 숨김 타이머
  setTimeout(() => {
    toast.classList.remove("show");
  }, 3000);
}

/**
 * 시간 형식 변환 (ISO 문자열 -> 현지 시간)
 * @param {string} isoString - ISO 형식 날짜 문자열
 * @returns {string} 현지 시간 문자열
 */
function formatLocalTime(isoString) {
  const date = new Date(isoString);
  return date.toLocaleString();
}

/**
 * 배치 API 테스트 실행
 * @param {Array} endpoints - 테스트할 엔드포인트 배열
 */
function runBatchTest(endpoints) {
  // 배치 테스트 로직 구현
  showToast(`${endpoints.length}개의 API 테스트를 시작합니다...`, "info");

  // 실제 구현은 필요에 따라 추가
}

/**
 * 데모 데이터 생성 (개발용)
 */
function generateDemoData() {
  // 개발 및 테스트용 데모 데이터 생성
  console.log("데모 데이터 생성 중...");
}

/**
 * 매거진 상태 자동 확인 (폴링)
 * @param {string} magazineId - 매거진 ID
 * @param {number} maxAttempts - 최대 시도 횟수
 */
function startMagazineStatusPolling(magazineId, maxAttempts) {
  let attempts = 0;
  const statusBox = document.getElementById("magazine-status-response");
  const intervalId = setInterval(async () => {
    attempts++;

    if (attempts > maxAttempts) {
      clearInterval(intervalId);
      showToast("매거진 상태 확인 시간이 초과되었습니다.", "warning");
      return;
    }

    try {
      const response = await fetch(`/magazine/status/${magazineId}`);
      const data = await response.json();

      // 상태 응답 박스에 표시
      if (statusBox) {
        statusBox.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
      }

      // 처리 완료되었거나 오류 발생 시 폴링 중지
      if (
        data.status === "completed" ||
        data.status === "failed" ||
        data.error
      ) {
        clearInterval(intervalId);

        if (data.status === "completed") {
          showToast("매거진 생성이 완료되었습니다!", "success");
        } else {
          showToast(
            `매거진 생성 오류: ${data.error || "알 수 없는 오류"}`,
            "error"
          );
        }
      }
    } catch (error) {
      console.error("매거진 상태 확인 중 오류:", error);
    }
  }, 5000); // 5초마다 확인
}

/**
 * 국가별 게시글 수 차트 렌더링
 * @param {Array} data - 국가별 게시글 수 데이터
 */
function renderCountryChart(data) {
  // 기존 차트 제거
  const chartContainer = document.getElementById("country-chart-container");
  const canvas = document.getElementById("countryChart");

  if (window.countryChart) {
    window.countryChart.destroy();
  }

  // 데이터 정렬 (게시글 수 기준 내림차순)
  const sortedData = [...data].sort((a, b) => b.count - a.count);

  // 차트 데이터 준비
  const countries = sortedData.map((item) => item.country);
  const counts = sortedData.map((item) => item.count);

  // 색상 배열 생성
  const backgroundColors = sortedData.map((_, index) => {
    const hue = (index * 30) % 360; // 색상 간격을 30도로 설정
    return `hsla(${hue}, 70%, 60%, 0.7)`;
  });

  const borderColors = backgroundColors.map((color) =>
    color.replace("0.7", "1")
  );

  // 차트 생성
  window.countryChart = new Chart(canvas.getContext("2d"), {
    type: "bar",
    data: {
      labels: countries,
      datasets: [
        {
          label: "게시글 수",
          data: counts,
          backgroundColor: backgroundColors,
          borderColor: borderColors,
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          grid: {
            color: darkMode ? "rgba(255, 255, 255, 0.1)" : "rgba(0, 0, 0, 0.1)",
          },
          ticks: {
            color: darkMode ? "#9ca3af" : "#6b7280",
            stepSize: 1,
            precision: 0,
          },
        },
        x: {
          grid: {
            display: false,
          },
          ticks: {
            color: darkMode ? "#9ca3af" : "#6b7280",
          },
        },
      },
      plugins: {
        legend: {
          labels: {
            color: darkMode ? "#f3f4f6" : "#1f2937",
          },
        },
        tooltip: {
          callbacks: {
            label: function (context) {
              return `게시글 수: ${context.raw}개`;
            },
          },
        },
      },
    },
  });

  showToast("국가별 게시글 수 차트가 업데이트되었습니다.", "success");
}

/**
 * 이미지 갤러리 렌더링
 * @param {Array} images - 이미지 목록 데이터
 * @param {string} containerId - 갤러리를 표시할 컨테이너 ID
 */
function renderImageGallery(images, containerId) {
  const container = document.getElementById(containerId);

  if (!images || images.length === 0) {
    container.innerHTML =
      '<div class="alert alert-info">이미지가 없습니다.</div>';
    return;
  }

  // JSON 응답 유지하되 아래에 갤러리 추가
  const jsonResponse = container.innerHTML;

  // 갤러리 HTML 생성
  let galleryHtml = '<div class="image-gallery" style="margin-top: 20px;">';
  galleryHtml += "<h4>이미지 갤러리</h4>";
  galleryHtml +=
    '<div class="gallery-grid" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px;">';

  images.forEach((image) => {
    galleryHtml += `
      <div class="image-item" style="border: 1px solid var(--border-color); border-radius: var(--radius); overflow: hidden;">
        <img src="${image.url}" alt="${image.name}" style="width: 100%; height: 120px; object-fit: cover;" />
        <div class="image-info" style="padding: 8px; font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
          ${image.name}
        </div>
        <div class="image-actions" style="padding: 0 8px 8px; display: flex; justify-content: space-between;">
          <button 
            class="copy-filename-btn" 
            style="font-size: 12px; padding: 3px 6px; border: none; background: var(--primary-color); color: white; border-radius: 3px; cursor: pointer;"
            data-filename="${image.name}">파일명 복사</button>
          <button 
            class="delete-image-btn" 
            style="font-size: 12px; padding: 3px 6px; border: none; background: var(--danger-color); color: white; border-radius: 3px; cursor: pointer;"
            data-filename="${image.name}">삭제</button>
        </div>
      </div>
    `;
  });

  galleryHtml += "</div></div>";

  // 결합 및 표시
  container.innerHTML = jsonResponse + galleryHtml;

  // 이벤트 리스너 추가
  const copyButtons = container.querySelectorAll(".copy-filename-btn");
  copyButtons.forEach((btn) => {
    btn.addEventListener("click", function () {
      const filename = this.getAttribute("data-filename");
      navigator.clipboard
        .writeText(filename)
        .then(() => {
          showToast(`파일명 '${filename}' 복사됨`, "success");
        })
        .catch((err) => {
          showToast(`복사 실패: ${err}`, "error");
        });
    });
  });

  const deleteButtons = container.querySelectorAll(".delete-image-btn");
  deleteButtons.forEach((btn) => {
    btn.addEventListener("click", function () {
      const filename = this.getAttribute("data-filename");
      const magazineId = document.getElementById(
        "storage-images-magazineID"
      ).value;

      // 삭제 확인
      if (confirm(`이미지 '${filename}'을(를) 삭제하시겠습니까?`)) {
        // 삭제 필드 자동 입력
        document.getElementById("storage-images-delete-magazineID").value =
          magazineId;
        document.getElementById("storage-images-delete-filename").value =
          filename;

        // 삭제 버튼 클릭
        document
          .querySelector('[data-endpoint="storage-images-delete"]')
          .click();
      }
    });
  });
}

/**
 * 오디오 플레이어 렌더링
 * @param {string} audioDataUri - 오디오 데이터 URI
 * @param {string} containerId - 플레이어를 표시할 컨테이너 ID
 */
function renderAudioPlayer(audioDataUri, containerId) {
  const container = document.getElementById(containerId);

  if (!audioDataUri) {
    container.innerHTML =
      '<div class="alert alert-warning">오디오 데이터가 없습니다.</div>';
    return;
  }

  // 오디오 플레이어 생성
  container.innerHTML = `
    <div class="audio-player" style="background-color: var(--background-secondary); border-radius: var(--radius); padding: 10px; margin-bottom: 10px;">
      <audio controls style="width: 100%;">
        <source src="${audioDataUri}" type="audio/mpeg">
        브라우저가 오디오 재생을 지원하지 않습니다.
      </audio>
      <div style="display: flex; justify-content: flex-end; margin-top: 5px;">
        <a href="${audioDataUri}" download="tts_audio.mp3" style="text-decoration: none; color: var(--primary-color); font-size: 0.875rem;">
          다운로드
        </a>
      </div>
    </div>
  `;

  // 자동 재생 (옵션)
  const audioElement = container.querySelector("audio");
  if (audioElement) {
    // 자동 재생은 사용자 설정에 따라 제한될 수 있음
    audioElement.play().catch((e) => console.log("Auto-play prevented:", e));
  }
}
