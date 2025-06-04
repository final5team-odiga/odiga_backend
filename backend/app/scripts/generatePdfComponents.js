// scripts/generatePdfComponents_AST.js
const fs = require("fs-extra");
const path = require("path");
const babelParser = require("@babel/parser");
const traverse = require("@babel/traverse").default;
const recast = require("recast");

const inputDir = path.join(__dirname, "../output/latest/components");
const outputDir = path.join(__dirname, "../pdf_components");

function extractText(node) {
  if (!node) return "";
  if (node.type === "JSXText") return node.value.trim();
  return "";
}

function extractAttr(node, attrName) {
  const attr = node.openingElement.attributes.find(
    (a) => a.name && a.name.name === attrName
  );
  return attr && attr.value && attr.value.value;
}

async function main() {
  await fs.ensureDir(outputDir);
  const files = await fs.readdir(inputDir);

  for (const file of files) {
    if (!file.endsWith(".jsx")) continue;

    const content = await fs.readFile(path.join(inputDir, file), "utf-8");
    const ast = babelParser.parse(content, {
      sourceType: "module",
      plugins: ["jsx"]
    });

    let title = "Untitled";
    let imageUrl = "https://via.placeholder.com/800x600";
    let body = "내용 없음";

    traverse(ast, {
      JSXElement(path) {
        const tag = path.node.openingElement.name.name;
        if (tag === "h2") {
          title = extractText(path.node.children[0]);
        }
        if (tag === "img") {
          imageUrl = extractAttr(path.node, "src") || imageUrl;
        }
        if (tag === "p") {
          body = extractText(path.node.children[0]) || body;
        }
      }
    });

    const componentName = path.basename(file, ".jsx") + "_PDF";
    const jsxCode = `
import React from 'react';

const ${componentName} = () => (
  <section style={{ padding: '40px', margin: '40px 0' }}>
    <h2 style={{ fontFamily: "'Noto Sans KR', sans-serif" }}>${title}</h2>
    <img
      src="${imageUrl}"
      alt="Static visual"
      style={{ width: '100%', borderRadius: '8px', margin: '20px 0' }}
    />
    <p style={{ fontFamily: "'Noto Sans KR', sans-serif", lineHeight: '1.6' }}>${body}</p>
  </section>
);

export default ${componentName};
`;

    const outputPath = path.join(outputDir, `${componentName}.jsx`);
    await fs.writeFile(outputPath, jsxCode, "utf-8");
    console.log(`✅ 변환 완료: ${outputPath}`);
  }
}

main().catch(console.error);
