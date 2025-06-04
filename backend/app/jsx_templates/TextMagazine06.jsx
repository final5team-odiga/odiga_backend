import React from "react";

const TextMagazine06 = () => {
  return (
    <div style={{ backgroundColor: "white", color: "black", padding: "20px" }}>
      <div
        style={{
          borderLeft: "4px solid black",
          paddingLeft: "20px",
          marginBottom: "32px",
        }}
      >
        <h1 style={{ fontSize: "2rem", marginBottom: "8px" }}>
          The Art of Minimalism
        </h1>
        <h2 style={{ fontSize: "1.125rem", marginBottom: "16px" }}>
          Less is more in design and life
        </h2>
      </div>

      <div style={{ marginBottom: "24px" }}>
        <h3 style={{ fontSize: "1.25rem", marginBottom: "8px" }}>
          01. Simplicity
        </h3>
        <p style={{ fontSize: "1rem", lineHeight: 1.5, marginLeft: "20px" }}>
          Minimalism strips away the unnecessary to reveal the essential. In a
          world of constant noise and distraction, simplicity becomes a refuge.
        </p>
      </div>

      <div style={{ marginBottom: "24px" }}>
        <h3 style={{ fontSize: "1.25rem", marginBottom: "8px" }}>
          02. Functionality
        </h3>
        <p style={{ fontSize: "1rem", lineHeight: 1.5, marginLeft: "20px" }}>
          Every element serves a purpose. Nothing exists merely for decoration.
          Form follows function in the purest sense.
        </p>
      </div>

      <div style={{ marginBottom: "24px" }}>
        <h3 style={{ fontSize: "1.25rem", marginBottom: "8px" }}>
          03. Clarity
        </h3>
        <p style={{ fontSize: "1rem", lineHeight: 1.5, marginLeft: "20px" }}>
          When we remove the clutter, what remains is clear, focused, and
          powerful. Minimalism is not about having less; it's about making room
          for what matters most.
        </p>
      </div>
    </div>
  );
};

export default TextMagazine06;
