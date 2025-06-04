import React from "react";

const ImageMagazine02 = () => {
  return (
    <div style={{ backgroundColor: "white", color: "black", padding: "20px" }}>
      <h1 style={{ fontSize: "2rem", marginBottom: "12px" }}>
        Urban Exploration
      </h1>
      <div style={{ display: "flex", gap: "8px" }}>
        <img
          src="https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=600"
          alt=""
          style={{ width: "50%", height: "auto", display: "block" }}
        />
        <img
          src="https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=600"
          alt=""
          style={{ width: "50%", height: "auto", display: "block" }}
        />
      </div>
      <h2 style={{ fontSize: "1.25rem", marginTop: "12px" }}>
        Discovering city life
      </h2>
    </div>
  );
};

export default ImageMagazine02;
