import React from "react";

const TextMagazine03 = () => {
  return (
    <div style={{ backgroundColor: "white", color: "black", padding: "20px" }}>
      <div style={{ display: "flex", gap: "24px" }}>
        <div style={{ flex: 1 }}>
          <h1 style={{ fontSize: "2rem", marginBottom: "16px" }}>
            Climate Change
          </h1>
          <p
            style={{ fontSize: "1rem", lineHeight: 1.5, marginBottom: "16px" }}
          >
            The Earth's climate is changing at an unprecedented rate. Rising
            temperatures, melting ice caps, and extreme weather events are
            becoming increasingly common.
          </p>
          <p style={{ fontSize: "1rem", lineHeight: 1.5 }}>
            Scientists worldwide are working tirelessly to understand and
            mitigate these changes through innovative research and sustainable
            solutions.
          </p>
        </div>
        <div style={{ flex: 1 }}>
          <h2 style={{ fontSize: "1.5rem", marginBottom: "16px" }}>
            Taking Action
          </h2>
          <p
            style={{ fontSize: "1rem", lineHeight: 1.5, marginBottom: "16px" }}
          >
            Individual actions, while seemingly small, can collectively make a
            significant impact. From reducing energy consumption to supporting
            renewable energy initiatives.
          </p>
          <p style={{ fontSize: "1rem", lineHeight: 1.5 }}>
            The time for action is now. Every choice we make today will
            determine the world we leave for future generations.
          </p>
        </div>
      </div>
    </div>
  );
};

export default TextMagazine03;
