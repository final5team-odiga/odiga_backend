import React from "react";

const ImageMagazine05 = () => {
  return (
    <div
      style={{
        position: "relative",
        backgroundColor: "white",
        color: "black",
        padding: "0",
      }}
    >
      <img
        src="https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1200"
        alt=""
        style={{ width: "100%", height: "auto", display: "block" }}
      />
      <div
        style={{
          position: "absolute",
          bottom: "20px",
          left: "20px",
          color: "black",
          backgroundColor: "rgba(255,255,255,0.7)",
          padding: "8px 12px",
          maxWidth: "60%",
        }}
      >
        <h1 style={{ fontSize: "2rem", margin: 0 }}>Sunset Serenity</h1>
        <h2 style={{ fontSize: "1.25rem", margin: 0 }}>
          Peaceful moments by the sea
        </h2>
      </div>
    </div>
  );
};

export default ImageMagazine05;
