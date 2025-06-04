import React from "react";

const MagazineArticle01 = () => {
  return (
    <div style={{ backgroundColor: "white", color: "black", padding: "24px" }}>
      <div style={{ display: "flex", gap: "20px", marginBottom: "20px" }}>
        <div style={{ flex: 2 }}>
          <h1
            style={{
              fontSize: "2.5rem",
              marginBottom: "8px",
              fontWeight: "bold",
            }}
          >
            The Future of Travel
          </h1>
          <h2
            style={{ fontSize: "1.25rem", marginBottom: "16px", color: "#666" }}
          >
            How technology is reshaping the way we explore
          </h2>
          <p
            style={{ fontSize: "1rem", lineHeight: 1.6, marginBottom: "16px" }}
          >
            The travel industry stands at the cusp of a technological
            revolution. From AI-powered trip planning to virtual reality
            previews of destinations, the way we discover and experience new
            places is evolving rapidly.
          </p>
          <p style={{ fontSize: "1rem", lineHeight: 1.6 }}>
            Smart luggage, real-time translation apps, and augmented reality
            city guides are just the beginning of what promises to be the most
            exciting era in travel history.
          </p>
        </div>
        <div style={{ flex: 1 }}>
          <img
            src="https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400"
            alt=""
            style={{ width: "100%", height: "auto", display: "block" }}
          />
        </div>
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: "12px",
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
    </div>
  );
};

export default MagazineArticle01;
