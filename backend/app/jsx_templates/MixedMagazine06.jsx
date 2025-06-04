import React from "react";

const MixedMagazine06 = () => {
  return (
    <div style={{ backgroundColor: "white", color: "black", padding: "20px" }}>
      <div style={{ display: "flex", gap: "20px", marginBottom: "24px" }}>
        <div style={{ flex: 1 }}>
          <img
            src="https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400"
            alt=""
            style={{ width: "100%", height: "auto", display: "block" }}
          />
        </div>
        <div style={{ flex: 2 }}>
          <h1 style={{ fontSize: "2rem", marginBottom: "12px" }}>
            Wildlife Conservation
          </h1>
          <h2 style={{ fontSize: "1.25rem", marginBottom: "16px" }}>
            Protecting our planet's biodiversity
          </h2>
          <p style={{ fontSize: "1rem", lineHeight: 1.5 }}>
            Every species plays a crucial role in maintaining the delicate
            balance of our ecosystems. Conservation efforts worldwide work
            tirelessly to protect endangered species and preserve their natural
            habitats for future generations.
          </p>
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: "12px",
          marginBottom: "20px",
        }}
      >
        <img
          src="https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=300"
          alt=""
          style={{ width: "100%", height: "auto", display: "block" }}
        />
        <img
          src="https://images.unsplash.com/photo-1439066615861-d1af74d74000?w=300"
          alt=""
          style={{ width: "100%", height: "auto", display: "block" }}
        />
        <img
          src="https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=300"
          alt=""
          style={{ width: "100%", height: "auto", display: "block" }}
        />
      </div>

      <p style={{ fontSize: "1rem", lineHeight: 1.5, textAlign: "center" }}>
        Through education, research, and direct action, we can ensure that the
        natural world continues to thrive alongside human civilization.
      </p>
    </div>
  );
};

export default MixedMagazine06;
