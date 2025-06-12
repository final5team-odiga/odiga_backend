require("@babel/register")({
  presets: ["@babel/preset-react"],
  extensions: [".jsx", ".js"],
  ignore: [/node_modules/],
});

const puppeteer = require("puppeteer");
const React = require("react");
const ReactDOM = require("react-dom/server");
const path = require("path");

/* ───── CLI ───── */
const args = process.argv.slice(2);
const iF = args.indexOf("--files"),
  iO = args.indexOf("--output");

if (iF === -1 || iO === -1 || iO <= iF) {
  console.error(
    "usage: node export_pdf.js --files <*.jsx …> --output <output.pdf>"
  );
  process.exit(1);
}

const jsxFiles = args.slice(iF + 1, iO);
const outPath = args[iO + 1];

/* ───── util ───── */
const GLOBAL_STYLES = `
<style>
  body { font-family: 'Noto Sans KR', sans-serif; line-height: 1.6; }
  img { max-width: 100%; height: auto; }
  .image-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; margin: 16px 0; }
  .image-container { border-radius: 8px; overflow: hidden; }
</style>
`;

const toHTML = (sections) => `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  ${GLOBAL_STYLES}
</head>
<body>
  ${sections.join("")}
</body>
</html>`;

/* ───── main ───── */
(async () => {
  try {
    const html = toHTML(
      jsxFiles.map((f) => {
        const C = require(path.resolve(f)).default || require(f);
        return ReactDOM.renderToStaticMarkup(React.createElement(C));
      })
    );

    const browser = await puppeteer.launch({
      headless: true,
      args: ["--no-sandbox", "--disable-setuid-sandbox"],
    });

    const page = await browser.newPage();
    await page.setViewport({ width: 1123, height: 1587, deviceScaleFactor: 1 });
    await page.emulateMediaType("screen");
    await page.setContent(html, { waitUntil: "networkidle0" });

    /* ✅ 폴백 이미지 완전 제거 */
    await page.addScriptTag({
      content: `
        (function(){
          // ✅ 존재하지 않는 투명 이미지로 대체
          const TRANSPARENT_IMG = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg'%3E%3C/svg%3E";
          
          function handleBrokenImage(img) {
            // ✅ 깨진 이미지는 완전히 숨김 처리
            img.style.display = 'none';
            img.style.visibility = 'hidden';
            img.style.opacity = '0';
            img.style.height = '0';
            img.style.width = '0';
          }
          
          document.querySelectorAll("img").forEach(img => {
            img.addEventListener("error", () => handleBrokenImage(img));
            
            // 이미 로드 실패한 이미지 처리
            if (!img.complete || img.naturalWidth === 0) {
              handleBrokenImage(img);
            }
          });
        })();`,
    });

    // 이미지 로딩 대기
    await page.evaluate(() =>
      Promise.all(
        [...document.images].map((img) =>
          img.complete ? 0 : new Promise((r) => (img.onload = img.onerror = r))
        )
      )
    );

    await page.pdf({
      path: outPath,
      format: "A4",
      printBackground: true,
      margin: { top: "20mm", bottom: "20mm", left: "15mm", right: "15mm" },
    });

    await browser.close();
    console.log("✅ PDF 저장:", outPath);
  } catch (e) {
    console.error("❌ PDF 생성 실패:", e);
    process.exit(1);
  }
})();
