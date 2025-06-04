import React from "react";

const MixedMagazine04 = () => {
  return (
    <div style={{ backgroundColor: "white", color: "black", padding: "20px" }}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "20px",
          marginBottom: "20px",
        }}
      >
        <img
          src="https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=500"
          alt=""
          style={{ width: "100%", height: "auto", display: "block" }}
        />
        <img
          src="https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=500"
          alt=""
          style={{ width: "100%", height: "auto", display: "block" }}
        />
      </div>
      <h1 style={{ fontSize: "2rem", marginBottom: "12px" }}>
        Seasonal Changes
      </h1>
      <h2 style={{ fontSize: "1.25rem", marginBottom: "16px" }}>
        Nature's eternal cycle
      </h2>
      <div
        style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px" }}
      >
        <p style={{ fontSize: "1rem", lineHeight: 1.5 }}>
          Spring brings renewal and growth, as dormant life awakens from
          winter's slumber. Buds burst forth with vibrant green leaves, and
          flowers paint the landscape in brilliant colors.
        </p>
        <p style={{ fontSize: "1rem", lineHeight: 1.5 }}>
          Summer's warmth nurtures life to its fullest expression, while
          autumn's golden hues remind us of nature's fleeting beauty. Winter's
          quiet reflection prepares the earth for another cycle of renewal.
        </p>
      </div>
    </div>
  );
};

export default MixedMagazine04;
