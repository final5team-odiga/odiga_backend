import React from "react";

const ImageMagazine04 = () => {
  return (
    <div style={{ backgroundColor: "white", color: "black", padding: "20px" }}>
      <div style={{ display: "flex", gap: "12px" }}>
        <div style={{ flex: 2 }}>
          <img
            src="https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800"
            alt=""
            style={{ width: "100%", height: "auto", display: "block" }}
          />
        </div>
        <div
          style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            gap: "12px",
          }}
        >
          <img
            src="https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=400"
            alt=""
            style={{ width: "100%", height: "auto", display: "block" }}
          />
          <img
            src="https://images.unsplash.com/photo-1439066615861-d1af74d74000?w=400"
            alt=""
            style={{ width: "100%", height: "auto", display: "block" }}
          />
        </div>
      </div>
      <h1 style={{ fontSize: "2rem", marginTop: "16px", marginBottom: "8px" }}>
        Mountain and Forest
      </h1>
      <h2 style={{ fontSize: "1.25rem", marginBottom: "12px" }}>
        Nature's Majesty
      </h2>
    </div>
  );
};

export default ImageMagazine04;
