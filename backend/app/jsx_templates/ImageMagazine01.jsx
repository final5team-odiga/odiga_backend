import React from "react";

const ImageMagazine01 = () => {
  return (
    <div style={{ backgroundColor: "white", color: "black", padding: "20px" }}>
      <div>
        <img
          src="https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1200"
          alt=""
          style={{ width: "100%", height: "auto", display: "block" }}
        />
      </div>
      <h1
        style={{ fontSize: "2.5rem", marginTop: "16px", marginBottom: "8px" }}
      >
        The Beauty of Nature
      </h1>
      <h2 style={{ fontSize: "1.5rem", marginBottom: "12px" }}>
        Exploring the great outdoors
      </h2>
      <p style={{ fontSize: "1rem", lineHeight: 1.5 }}>
        Experience the breathtaking landscapes and serene environments that
        nature offers.
      </p>
    </div>
  );
};

export default ImageMagazine01;
