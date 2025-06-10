require("@babel/register")({
  presets: ["@babel/preset-react"],
  extensions: [".jsx", ".js"],
  ignore: [/node_modules/],
});

const puppeteer = require("puppeteer");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const fs = require("fs");
const path = require("path");

const args = process.argv.slice(2);
const fileArgIndex = args.indexOf("--files");
const outputArgIndex = args.indexOf("--output");

if (
  fileArgIndex === -1 ||
  outputArgIndex === -1 ||
  outputArgIndex <= fileArgIndex
) {
  console.error("❌ 명령줄 인자 오류: --files 와 --output 순서를 확인하세요.");
  process.exit(1);
}

const jsxFiles = args.slice(fileArgIndex + 1, outputArgIndex);
const outputPath = args[outputArgIndex + 1];

// ✅ 완전히 개선된 HTML 템플릿 (레이아웃 보존)
const createFullHTML = (htmlSections) => `
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Magazine PDF</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap" rel="stylesheet">
  <style>
    /* ✅ 페이지 미디어 설정 */
    @page {
      size: A4;
      margin: 8mm;
    }
    
    /* ✅ 프린트 미디어 전용 스타일 - 레이아웃 보존 */
    @media print {
      body {
        margin: 0;
        padding: 0;
        font-family: 'Noto Sans KR', sans-serif;
        line-height: 1.6;
        color: black !important;
        background: white !important;
        font-size: 11pt;
      }
      
      /* ✅ 페이지 분할 제어 */
      .page-section {
        page-break-after: always;
        page-break-inside: avoid;
        min-height: 85vh;
        max-height: 95vh;
        position: relative;
        overflow: hidden;
        padding: 8mm;
        box-sizing: border-box;
      }
      
      .page-section:last-child {
        page-break-after: auto;
      }
      
      /* ✅ Flexbox 레이아웃 보존 (조건부 비활성화) */
      .flex {
        display: flex !important;
      }
      
      /* ✅ 특정 Flexbox만 비활성화 (페이지 분할 문제가 있는 경우) */
      .flex.print-block {
        display: block !important;
      }
      
      .flex.print-block > * {
        display: block !important;
        margin-bottom: 8pt;
        width: 100% !important;
      }
      
      /* ✅ Grid 레이아웃 보존 */
      .grid {
        display: grid !important;
      }
      
      .grid.print-block {
        display: block !important;
      }
      
      .grid.print-block > * {
        display: block !important;
        margin-bottom: 8pt;
        width: 100% !important;
      }
      
      /* ✅ 이미지 크기 제어 (레이아웃 보존) */
      img {
        max-width: 100% !important;
        height: auto !important;
        max-height: 35vh !important;
        object-fit: contain !important;
        page-break-inside: avoid;
      }
      
      /* ✅ Flexbox 내 이미지는 inline-block 유지 */
      .flex img {
        display: inline-block !important;
        margin: 4pt !important;
      }
      
      /* ✅ Grid 내 이미지는 block 유지하되 크기 제한 */
      .grid img {
        display: block !important;
        margin: 4pt auto !important;
        max-width: 200pt !important;
        max-height: 150pt !important;
      }
      
      /* ✅ 단독 이미지는 중앙 정렬 */
      img:only-child, .image-container img {
        display: block !important;
        margin: 8pt auto !important;
        max-width: 300pt !important;
        max-height: 200pt !important;
      }
      
      /* ✅ 텍스트 요소 최적화 */
      h1, h2, h3, h4, h5, h6 {
        page-break-after: avoid;
        page-break-inside: avoid;
        margin-bottom: 6pt !important;
        margin-top: 8pt !important;
        color: black !important;
        font-weight: bold !important;
      }
      
      h1 { font-size: 16pt !important; line-height: 1.2 !important; }
      h2 { font-size: 13pt !important; line-height: 1.3 !important; }
      h3 { font-size: 11pt !important; line-height: 1.4 !important; }
      
      p {
        page-break-inside: avoid;
        margin-bottom: 6pt !important;
        margin-top: 0 !important;
        text-align: justify !important;
        hyphens: auto;
        word-wrap: break-word;
        font-size: 10pt !important;
        line-height: 1.5 !important;
      }
      
      /* ✅ 컨테이너 최적화 */
      .container, .max-w-4xl, .max-w-3xl, [class*="max-w"] {
        max-width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
        width: 100% !important;
      }
      
      /* ✅ 여백 최적화 */
      .p-8, .p-6, .p-4 { padding: 6pt !important; }
      .mb-8, .mb-6, .mb-4 { margin-bottom: 6pt !important; }
      .mt-8, .mt-6, .mt-4 { margin-top: 6pt !important; }
      .gap-8, .gap-6, .gap-4 { gap: 4pt !important; }
      
      /* ✅ 원형 이미지 처리 */
      .rounded-full {
        border-radius: 6pt !important;
        max-width: 80pt !important;
        max-height: 80pt !important;
      }
      
      /* ✅ 그림자 제거 */
      .shadow-lg, .shadow-md, .shadow, [class*="shadow"] {
        box-shadow: none !important;
      }
      
      /* ✅ 색상 최적화 */
      .text-white { color: black !important; }
      .bg-black { background-color: white !important; }
      
      /* ✅ 특정 템플릿 레이아웃 보존 */
      .template-flex-preserve {
        display: flex !important;
        flex-direction: row !important;
        align-items: flex-start !important;
        gap: 12pt !important;
      }
      
      .template-flex-preserve > * {
        flex: 1 !important;
      }
      
      .template-grid-preserve {
        display: grid !important;
        grid-template-columns: repeat(auto-fit, minmax(120pt, 1fr)) !important;
        gap: 8pt !important;
      }
    }
    
    /* ✅ 스크린 미디어 (개발 시 확인용) */
    @media screen {
      body {
        font-family: 'Noto Sans KR', sans-serif;
        line-height: 1.6;
      }
      
      .page-section {
        border: 1px dashed #ccc;
        margin-bottom: 2rem;
        padding: 1rem;
        min-height: 400px;
      }
    }
  </style>
</head>
<body>
  ${htmlSections
    .map(
      (section, index) => `
    <div class="page-section">
      ${section}
    </div>
  `
    )
    .join("")}
</body>
</html>`;

(async () => {
  const browser = await puppeteer.launch({
    headless: true,
    args: [
      "--no-sandbox",
      "--disable-setuid-sandbox",
      "--disable-dev-shm-usage",
      "--disable-gpu",
      "--disable-features=VizDisplayCompositor",
      "--font-render-hinting=none", // ✅ 폰트 렌더링 최적화
    ],
  });

  const page = await browser.newPage();

  // ✅ 프린트 미디어 에뮬레이션 (필수!)
  await page.emulateMediaType("print");

  // ✅ 이미지 요청 인터셉션 및 최적화 (검색 결과 6번 참조)
  await page.setRequestInterception(true);
  page.on("request", async (req) => {
    if (req.resourceType() !== "image") {
      req.continue();
      return;
    }

    try {
      const response = await fetch(req.url(), {
        method: req.method(),
        headers: req.headers(),
      });

      if (response.ok) {
        const buffer = await response.arrayBuffer();

        // ✅ 이미지 크기 제한 및 최적화
        const sharp = require("sharp");
        const optimizedImage = await sharp(buffer)
          .resize(800, 600, {
            fit: "inside",
            withoutEnlargement: true,
          })
          .jpeg({
            quality: 80,
            mozjpeg: true,
          })
          .rotate() // EXIF 메타데이터 기반 회전
          .toBuffer();

        req.respond({
          body: optimizedImage,
          headers: {
            "Content-Type": "image/jpeg",
          },
        });
      } else {
        req.continue();
      }
    } catch (error) {
      console.warn(`이미지 최적화 실패: ${req.url()}`);
      req.continue();
    }
  });

  const htmlSections = [];

  // JSX 컴포넌트들을 HTML로 변환
  for (const jsxPath of jsxFiles) {
    try {
      const componentPath = path.resolve(jsxPath);
      delete require.cache[componentPath];

      const mod = require(componentPath);
      const Component = mod.default || mod;

      if (!Component) {
        throw new Error(
          "컴포넌트를 찾을 수 없습니다 (default export 누락 가능)."
        );
      }

      const html = ReactDOMServer.renderToStaticMarkup(
        React.createElement(Component)
      );

      htmlSections.push(html);
      console.log(`✅ 컴포넌트 변환 완료: ${path.basename(jsxPath)}`);
    } catch (err) {
      console.error(`❌ 컴포넌트 로딩 실패: ${jsxPath}`);
      console.error(err);
      await browser.close();
      process.exit(1);
    }
  }

  const finalHTML = createFullHTML(htmlSections);

  // ✅ HTML을 페이지에 설정
  await page.setContent(finalHTML, {
    waitUntil: "networkidle0",
    timeout: 60000,
  });

  // ✅ 최적화된 PDF 생성 옵션
  const pdfOptions = {
    path: outputPath,
    format: "A4",
    printBackground: true,
    margin: {
      top: "10mm",
      right: "10mm",
      bottom: "10mm",
      left: "10mm",
    },
    preferCSSPageSize: true,
    displayHeaderFooter: false,
    scale: 0.9, // ✅ 스케일 조정으로 내용이 페이지에 잘 맞도록
  };

  await page.pdf(pdfOptions);
  await browser.close();

  console.log(`✅ PDF 생성 완료: ${outputPath}`);
})();
