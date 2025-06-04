import React from "react";

const MagazineGuide01 = () => {
  return (
    <div style={{ backgroundColor: "white", color: "black", padding: "24px" }}>
      <h1
        style={{
          fontSize: "2.5rem",
          marginBottom: "12px",
          fontWeight: "bold",
          textAlign: "center",
        }}
      >
        48 Hours in Tokyo
      </h1>
      <h2
        style={{
          fontSize: "1.25rem",
          marginBottom: "24px",
          textAlign: "center",
          color: "#666",
        }}
      >
        The ultimate weekend itinerary
      </h2>

      <div style={{ display: "flex", gap: "20px", marginBottom: "24px" }}>
        <img
          src="https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=600"
          alt=""
          style={{ width: "50%", height: "auto", display: "block" }}
        />
        <img
          src="https://images.unsplash.com/photo-1439066615861-d1af74d74000?w=600"
          alt=""
          style={{ width: "50%", height: "auto", display: "block" }}
        />
      </div>

      <div
        style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px" }}
      >
        <div>
          <h3
            style={{
              fontSize: "1.5rem",
              marginBottom: "16px",
              fontWeight: "bold",
              color: "#333",
            }}
          >
            Day 1: Traditional Tokyo
          </h3>

          <div style={{ marginBottom: "16px" }}>
            <h4
              style={{
                fontSize: "1.125rem",
                marginBottom: "8px",
                fontWeight: "bold",
              }}
            >
              Morning (9:00 AM)
            </h4>
            <p
              style={{
                fontSize: "0.9rem",
                lineHeight: 1.4,
                marginBottom: "8px",
              }}
            >
              Start at Senso-ji Temple in Asakusa. Explore the traditional
              shopping street and sample local snacks like ningyo-yaki and melon
              pan.
            </p>
          </div>

          <div style={{ marginBottom: "16px" }}>
            <h4
              style={{
                fontSize: "1.125rem",
                marginBottom: "8px",
                fontWeight: "bold",
              }}
            >
              Afternoon (1:00 PM)
            </h4>
            <p
              style={{
                fontSize: "0.9rem",
                lineHeight: 1.4,
                marginBottom: "8px",
              }}
            >
              Visit the Imperial Palace East Gardens. Enjoy a traditional
              kaiseki lunch in nearby Ginza district.
            </p>
          </div>

          <div>
            <h4
              style={{
                fontSize: "1.125rem",
                marginBottom: "8px",
                fontWeight: "bold",
              }}
            >
              Evening (6:00 PM)
            </h4>
            <p style={{ fontSize: "0.9rem", lineHeight: 1.4 }}>
              Experience the vibrant nightlife of Shibuya. Don't miss the famous
              crossing and panoramic city views from Sky Gallery.
            </p>
          </div>
        </div>

        <div>
          <h3
            style={{
              fontSize: "1.5rem",
              marginBottom: "16px",
              fontWeight: "bold",
              color: "#333",
            }}
          >
            Day 2: Modern Tokyo
          </h3>

          <div style={{ marginBottom: "16px" }}>
            <h4
              style={{
                fontSize: "1.125rem",
                marginBottom: "8px",
                fontWeight: "bold",
              }}
            >
              Morning (10:00 AM)
            </h4>
            <p
              style={{
                fontSize: "0.9rem",
                lineHeight: 1.4,
                marginBottom: "8px",
              }}
            >
              Explore the trendy Harajuku district and Omotesando Hills. Perfect
              for fashion enthusiasts and architecture lovers.
            </p>
          </div>

          <div style={{ marginBottom: "16px" }}>
            <h4
              style={{
                fontSize: "1.125rem",
                marginBottom: "8px",
                fontWeight: "bold",
              }}
            >
              Afternoon (2:00 PM)
            </h4>
            <p
              style={{
                fontSize: "0.9rem",
                lineHeight: 1.4,
                marginBottom: "8px",
              }}
            >
              Visit teamLab Borderless digital art museum. Book tickets in
              advance for this immersive technological experience.
            </p>
          </div>

          <div>
            <h4
              style={{
                fontSize: "1.125rem",
                marginBottom: "8px",
                fontWeight: "bold",
              }}
            >
              Evening (7:00 PM)
            </h4>
            <p style={{ fontSize: "0.9rem", lineHeight: 1.4 }}>
              End with dinner in Roppongi Hills and night views from Tokyo Tower
              or Tokyo Skytree.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MagazineGuide01;
