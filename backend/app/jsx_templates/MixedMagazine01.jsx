import React from "react";

const MixedMagazine01 = () => {
  return (
    <div style={{ backgroundColor: "white", color: "black", padding: "20px" }}>
      <div style={{ display: "flex", gap: "20px", alignItems: "flex-start" }}>
        <div style={{ flex: 1 }}>
          <img
            src="https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=600"
            alt=""
            style={{ width: "100%", height: "auto", display: "block" }}
          />
        </div>
        <div style={{ flex: 1 }}>
          <h1 style={{ fontSize: "2rem", marginBottom: "12px" }}>
            Mountain Adventures
          </h1>
          <h2 style={{ fontSize: "1.25rem", marginBottom: "16px" }}>
            Discovering peaks and valleys
          </h2>
          <p
            style={{ fontSize: "1rem", lineHeight: 1.5, marginBottom: "12px" }}
          >
            The mountains call to those who seek adventure and solitude. Each
            peak offers a unique challenge and reward for those brave enough to
            climb.
          </p>
          <p style={{ fontSize: "1rem", lineHeight: 1.5 }}>
            From the first light of dawn breaking over snow-capped summits to
            the peaceful silence of alpine meadows, mountain adventures create
            memories that last a lifetime.
          </p>
        </div>
      </div>
    </div>
  );
};

export default MixedMagazine01;
