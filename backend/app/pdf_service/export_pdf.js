// export_pdf.js
require("@babel/register")({
  presets: ["@babel/preset-react"],
  extensions: [".jsx", ".js"],
  ignore: [/node_modules/]
});

const puppeteer = require("puppeteer");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const fs = require("fs");
const path = require("path");

const args = process.argv.slice(2);
const fileArgIndex = args.indexOf("--files");
const outputArgIndex = args.indexOf("--output");

if (fileArgIndex === -1 || outputArgIndex === -1 || outputArgIndex <= fileArgIndex) {
  console.error("❌ 명령줄 인자 오류: --files 와 --output 순서를 확인하세요.");
  process.exit(1);
}

const jsxFiles = args.slice(fileArgIndex + 1, outputArgIndex);
const outputPath = args[outputArgIndex + 1];

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();

  const htmlSections = [];

  for (const jsxPath of jsxFiles) {
    try {
      const componentPath = path.resolve(jsxPath);
      const mod = require(componentPath);
      const Component = mod.default || mod;

      if (!Component) {
        throw new Error("컴포넌트를 찾을 수 없습니다 (default export 누락 가능).");
      }

      const html = ReactDOMServer.renderToStaticMarkup(
        React.createElement(Component)
      );
      htmlSections.push(html);
    } catch (err) {
      console.error(`❌ 컴포넌트 로딩 실패: ${jsxPath}`);
      console.error(err);
      await browser.close();
      process.exit(1);
    }
  }

  const finalHTML = `
    <html>
      <head>
        <style>
          body { margin: 0; padding: 0; font-family: sans-serif; }
          .page-break { page-break-after: always; }
        </style>
      </head>
      <body>
        ${htmlSections.map(section => `<div class="page">${section}</div><div class="page-break"></div>`).join("")}
      </body>
    </html>
  `;

  try {
    await page.setContent(finalHTML, { waitUntil: "networkidle0" });
    await page.pdf({ path: outputPath, format: "A4" });
    console.log(`✅ PDF 생성 완료: ${outputPath}`);
  } catch (err) {
    console.error("❌ PDF 생성 중 오류 발생:");
    console.error(err);
    process.exit(1);
  }

  await browser.close();
})();
