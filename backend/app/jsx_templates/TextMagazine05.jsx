import React from "react";

const TextMagazine05 = () => {
  return (
    <div style={{ backgroundColor: "white", color: "black", padding: "20px" }}>
      <h1
        style={{ fontSize: "4rem", textAlign: "center", marginBottom: "8px" }}
      >
        INNOVATION
      </h1>
      <h2
        style={{
          fontSize: "1.5rem",
          textAlign: "center",
          marginBottom: "32px",
        }}
      >
        The driving force of progress
      </h2>

      <div style={{ textAlign: "center", marginBottom: "32px" }}>
        <p
          style={{
            fontSize: "1.25rem",
            lineHeight: 1.4,
            maxWidth: "600px",
            margin: "0 auto",
          }}
        >
          Innovation is not just about creating something new; it's about
          solving problems, improving lives, and pushing the boundaries of
          what's possible.
        </p>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: "24px",
          marginTop: "32px",
        }}
      >
        <div style={{ textAlign: "center" }}>
          <h3 style={{ fontSize: "1.25rem", marginBottom: "8px" }}>THINK</h3>
          <p style={{ fontSize: "0.9rem", lineHeight: 1.4 }}>
            Challenge conventional wisdom and explore new possibilities
          </p>
        </div>
        <div style={{ textAlign: "center" }}>
          <h3 style={{ fontSize: "1.25rem", marginBottom: "8px" }}>CREATE</h3>
          <p style={{ fontSize: "0.9rem", lineHeight: 1.4 }}>
            Transform ideas into tangible solutions that make a difference
          </p>
        </div>
        <div style={{ textAlign: "center" }}>
          <h3 style={{ fontSize: "1.25rem", marginBottom: "8px" }}>IMPACT</h3>
          <p style={{ fontSize: "0.9rem", lineHeight: 1.4 }}>
            Measure success by the positive change you bring to the world
          </p>
        </div>
      </div>
    </div>
  );
};

export default TextMagazine05;
