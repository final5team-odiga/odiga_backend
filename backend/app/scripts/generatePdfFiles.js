// scripts/generatePdfFiles.js
const fs = require("fs");
const path = require("path");
const { exec } = require("child_process");
const fse = require("fs-extra");

const COMPONENT_DIR = path.join(__dirname, "../pdf_components");
const OUTPUT_DIR = path.join(__dirname, "../output_pdfs");
const EXPORT_SCRIPT = path.join(__dirname, "../pdf_service/export_pdf.js");
const OUTPUT_PDF_PATH = path.join(OUTPUT_DIR, "magazine_result.pdf");

async function run() {
  await fse.ensureDir(OUTPUT_DIR);

  // ⛳ JSX 파일들 정렬
  const files = fs.readdirSync(COMPONENT_DIR)
    .filter((f) => f.endsWith(".jsx"))
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));

  if (files.length === 0) {
    console.error("❌ 변환할 JSX 파일이 없습니다.");
    return;
  }

  // ⛳ 전체 JSX 경로를 공백으로 이어붙여 인자로 넘김
  const jsxPaths = files.map((file) =>
    path.join(COMPONENT_DIR, file)
  ).join(" ");

  const command = `npx babel-node ${EXPORT_SCRIPT} --files ${jsxPaths} --output ${OUTPUT_PDF_PATH}`;
  console.log("📄 통합 PDF 생성 시작...");

  await new Promise((resolve, reject) => {
    exec(command, (err, stdout, stderr) => {
      if (err) {
        console.error("❌ PDF 변환 실패:");
        console.error(stderr);
        return reject(err);
      }
      console.log(`✅ PDF 생성 완료: ${OUTPUT_PDF_PATH}`);
      resolve();
    });
  });
}

run().catch((e) => {
  console.error("🚨 전체 처리 중단:", e);
});