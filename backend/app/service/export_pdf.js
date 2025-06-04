// pdf_service/export_pdf.js
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

// Tailwind CSS 포함한 완전한 HTML 템플릿
const createFullHTML = (htmlSections) => `
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Magazine PDF</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Source+Sans+Pro:wght@300;400;600&family=Montserrat:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    @page {
      size: A4;
      margin: 0;
    }
    body {
      margin: 0;
      padding: 0;
      font-family: 'Source Sans Pro', sans-serif;
      line-height: 1.6;
    }
    .page-break {
      page-break-after: always;
    }
    .no-break {
      page-break-inside: avoid;
    }
    img {
      max-width: 100%;
      height: auto;
    }
    /* Tailwind 폰트 패밀리 커스텀 */
    .font-playfair { font-family: 'Playfair Display', serif !important; }
    .font-source { font-family: 'Source Sans Pro', sans-serif !important; }
    .font-montserrat { font-family: 'Montserrat', sans-serif !important; }
  </style>
</head>
<body>
  ${htmlSections
    .map(
      (section, index) => `
    <div class="no-break ${
      index < htmlSections.length - 1 ? "page-break" : ""
    }">
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
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });

  const page = await browser.newPage();
  const htmlSections = [];

  // JSX 컴포넌트들을 HTML로 변환
  for (const jsxPath of jsxFiles) {
    try {
      const componentPath = path.resolve(jsxPath);

      // 캐시 클리어
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

  // HTML을 페이지에 설정
  await page.setContent(finalHTML, {
    waitUntil: "networkidle0",
    timeout: 30000,
  });

  // PDF 생성 옵션
  const pdfOptions = {
    path: outputPath,
    format: "A4",
    printBackground: true,
    margin: {
      top: "0.5in",
      right: "0.5in",
      bottom: "0.5in",
      left: "0.5in",
    },
    preferCSSPageSize: true,
  };

  await page.pdf(pdfOptions);
  await browser.close();

  console.log(`✅ PDF 생성 완료: ${outputPath}`);
})();
