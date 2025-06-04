import React from "react";

const ImageMagazine03 = () => {
  return (
    <div style={{ backgroundColor: "white", color: "black", padding: "20px" }}>
      <h1 style={{ fontSize: "2rem", marginBottom: "16px" }}>
        Seaside Adventures
      </h1>
      <div style={{ display: "flex", gap: "8px" }}>
        <img
          src="https://images.unsplash.com/photo-1439066615861-d1af74d74000?w=400"
          alt=""
          style={{ width: "33.33%", height: "auto", display: "block" }}
        />
        <img
          src="https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400"
          alt=""
          style={{ width: "33.33%", height: "auto", display: "block" }}
        />
        <img
          src="https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=400"
          alt=""
          style={{ width: "33.33%", height: "auto", display: "block" }}
        />
      </div>
    </div>
  );
};

export default ImageMagazine03;
