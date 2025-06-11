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
  <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&family=Playfair+Display:wght@400;700&family=Montserrat:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    /* ✅ 페이지 미디어 설정 */
    @page {
      size: A4;
      margin: 8mm;
    }
    
    /* ✅ 프린트 미디어 - 타이포그래피 중심 다양성 */
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
        background: white !important;
      }
      
      .page-section:last-child {
        page-break-after: auto;
      }
      
      /* ✅ 템플릿별 폰트 패밀리 다양성 */
      .template-elegant { 
        font-family: 'Playfair Display', serif !important; 
      }
      
      .template-modern { 
        font-family: 'Montserrat', sans-serif !important; 
      }
      
      .template-classic { 
        font-family: 'Noto Sans KR', sans-serif !important; 
      }
      
      /* ✅ 제목 스타일 다양성 (크기, 간격, 정렬) */
      .title-elegant { 
        font-size: 24pt !important; 
        font-family: 'Playfair Display', serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.8pt !important;
        line-height: 1.1 !important;
        margin-bottom: 12pt !important;
        text-align: left !important;
      }
      
      .title-modern { 
        font-size: 20pt !important; 
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 2pt !important;
        line-height: 1.2 !important;
        margin-bottom: 8pt !important;
        text-align: center !important;
      }
      
      .title-classic { 
        font-size: 18pt !important; 
        font-family: 'Noto Sans KR', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 0pt !important;
        line-height: 1.3 !important;
        margin-bottom: 10pt !important;
        text-align: left !important;
      }
      
      .title-minimal { 
        font-size: 16pt !important; 
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 300 !important;
        letter-spacing: 1pt !important;
        line-height: 1.4 !important;
        margin-bottom: 6pt !important;
        text-align: right !important;
      }
      
      /* ✅ 부제목 스타일 다양성 */
      .subtitle-elegant { 
        font-size: 14pt !important; 
        font-family: 'Playfair Display', serif !important;
        font-style: italic !important;
        font-weight: 400 !important;
        margin-bottom: 8pt !important;
        text-align: left !important;
      }
      
      .subtitle-modern { 
        font-size: 12pt !important; 
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 500 !important;
        text-transform: lowercase !important;
        letter-spacing: 0.5pt !important;
        margin-bottom: 6pt !important;
        text-align: center !important;
      }
      
      .subtitle-classic { 
        font-size: 13pt !important; 
        font-family: 'Noto Sans KR', sans-serif !important;
        font-weight: 500 !important;
        margin-bottom: 8pt !important;
        text-align: left !important;
      }
      
      /* ✅ 본문 텍스트 스타일 다양성 */
      .text-elegant { 
        font-size: 11pt !important; 
        font-family: 'Playfair Display', serif !important;
        line-height: 1.8 !important;
        text-align: justify !important;
        margin-bottom: 10pt !important;
        text-indent: 12pt !important;
      }
      
      .text-modern { 
        font-size: 10pt !important; 
        font-family: 'Montserrat', sans-serif !important;
        line-height: 1.6 !important;
        text-align: left !important;
        margin-bottom: 8pt !important;
        letter-spacing: 0.2pt !important;
      }
      
      .text-classic { 
        font-size: 10pt !important; 
        font-family: 'Noto Sans KR', sans-serif !important;
        line-height: 1.7 !important;
        text-align: justify !important;
        margin-bottom: 8pt !important;
      }
      
      /* ✅ 레이아웃 다양성 (Flexbox/Grid 보존) */
      .layout-elegant {
        display: flex !important;
        flex-direction: row !important;
        align-items: flex-start !important;
        gap: 16pt !important;
        padding: 16pt !important;
      }
      
      .layout-modern {
        display: grid !important;
        grid-template-columns: 1fr 1fr !important;
        gap: 12pt !important;
        padding: 12pt !important;
      }
      
      .layout-classic {
        display: block !important;
        padding: 14pt !important;
      }
      
      .layout-magazine {
        display: grid !important;
        grid-template-columns: 2fr 1fr !important;
        gap: 14pt !important;
        padding: 10pt !important;
      }
      
      .layout-gallery {
        display: grid !important;
        grid-template-columns: repeat(3, 1fr) !important;
        gap: 8pt !important;
        padding: 8pt !important;
      }
      
      /* ✅ 이미지 스타일 다양성 (크기와 배치) */
      .image-elegant {
        border-radius: 8pt !important;
        max-width: 100% !important;
        height: auto !important;
        max-height: 45vh !important;
        object-fit: cover !important;
        margin: 6pt !important;
      }
      
      .image-modern {
        border-radius: 2pt !important;
        max-width: 100% !important;
        height: auto !important;
        max-height: 35vh !important;
        object-fit: contain !important;
        margin: 4pt !important;
      }
      
      .image-classic {
        border-radius: 4pt !important;
        max-width: 100% !important;
        height: auto !important;
        max-height: 40vh !important;
        object-fit: cover !important;
        margin: 8pt auto !important;
        display: block !important;
      }
      
      .image-small {
        max-width: 150pt !important;
        max-height: 120pt !important;
        object-fit: cover !important;
        margin: 4pt !important;
      }
      
      .image-large {
        max-width: 100% !important;
        max-height: 50vh !important;
        object-fit: cover !important;
        margin: 8pt auto !important;
        display: block !important;
      }
      
      /* ✅ 특별한 레이아웃 요소들 (흰색 배경 내에서 구분) */
      .content-box-elegant {
        border-left: 3pt solid #ddd !important;
        padding-left: 12pt !important;
        margin: 10pt 0 !important;
      }
      
      .content-box-modern {
        border: 1pt solid #eee !important;
        padding: 10pt !important;
        margin: 8pt 0 !important;
      }
      
      .content-box-minimal {
        border-top: 1pt solid #f0f0f0 !important;
        padding-top: 8pt !important;
        margin-top: 12pt !important;
      }
      
      /* ✅ 템플릿별 고유 특성 */
      .template-mixed-07 {
        border-top: 2pt solid #e8e8e8 !important;
        padding-top: 10pt !important;
      }
      
      .template-mixed-08 {
        text-align: center !important;
        padding: 12pt 8pt !important;
      }
      
      .template-mixed-10 {
        border: 1pt solid #f5f5f5 !important;
        padding: 12pt !important;
        margin: 4pt !important;
      }
      
      .template-mixed-13 {
        display: grid !important;
        grid-template-rows: auto 1fr auto !important;
        min-height: 70vh !important;
      }
      
      /* ✅ 간격과 여백 다양성 */
      .spacing-tight {
        padding: 6pt !important;
        margin-bottom: 4pt !important;
      }
      
      .spacing-normal {
        padding: 10pt !important;
        margin-bottom: 8pt !important;
      }
      
      .spacing-loose {
        padding: 16pt !important;
        margin-bottom: 12pt !important;
      }
      
      /* ✅ 기본 요소 최적화 */
      h1, h2, h3, h4, h5, h6 {
        page-break-after: avoid;
        page-break-inside: avoid;
        color: black !important;
        font-weight: bold !important;
      }
      
      p {
        page-break-inside: avoid;
        color: black !important;
        hyphens: auto;
        word-wrap: break-word;
      }
      
      img {
        page-break-inside: avoid;
        color: black !important;
      }
      
      /* ✅ 컨테이너 최적화 */
      .container, .max-w-4xl, .max-w-3xl, [class*="max-w"] {
        max-width: 100% !important;
        width: 100% !important;
      }
    }
    
    /* ✅ 스크린 미디어 (개발 시 확인용) */
    @media screen {
      body {
        font-family: 'Noto Sans KR', sans-serif;
        line-height: 1.6;
        background: white;
        color: black;
      }
      
      .page-section {
        border: 1px dashed #ccc;
        margin-bottom: 2rem;
        padding: 1rem;
        min-height: 400px;
        background: white;
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
