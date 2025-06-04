import React from "react";

const MixedMagazine02 = () => {
  return (
    <div style={{ backgroundColor: "white", color: "black", padding: "20px" }}>
      <h1
        style={{
          fontSize: "2.5rem",
          marginBottom: "16px",
          textAlign: "center",
        }}
      >
        Ocean Mysteries
      </h1>
      <img
        src="https://images.unsplash.com/photo-1439066615861-d1af74d74000?w=1000"
        alt=""
        style={{
          width: "100%",
          height: "auto",
          display: "block",
          marginBottom: "20px",
        }}
      />
      <div style={{ display: "flex", gap: "24px" }}>
        <div style={{ flex: 1 }}>
          <h2 style={{ fontSize: "1.5rem", marginBottom: "12px" }}>
            Depths Unknown
          </h2>
          <p style={{ fontSize: "1rem", lineHeight: 1.5 }}>
            The ocean covers more than 70% of our planet's surface, yet we have
            explored less than 5% of its depths. Hidden beneath the waves lie
            countless mysteries waiting to be discovered.
          </p>
        </div>
        <div style={{ flex: 1 }}>
          <h2 style={{ fontSize: "1.5rem", marginBottom: "12px" }}>
            Marine Life
          </h2>
          <p style={{ fontSize: "1rem", lineHeight: 1.5 }}>
            From microscopic plankton to massive whales, the ocean teems with
            life in forms both familiar and alien. Each species plays a crucial
            role in maintaining the delicate balance of marine ecosystems.
          </p>
        </div>
      </div>
    </div>
  );
};

export default MixedMagazine02;
