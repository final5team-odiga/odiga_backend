import React from "react";

const MagazineInterview01 = () => {
  return (
    <div style={{ backgroundColor: "white", color: "black", padding: "24px" }}>
      <div style={{ display: "flex", gap: "24px", alignItems: "flex-start" }}>
        <div style={{ flex: 1 }}>
          <img
            src="https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=500"
            alt=""
            style={{ width: "100%", height: "auto", display: "block" }}
          />
        </div>
        <div style={{ flex: 2 }}>
          <div style={{ marginBottom: "20px" }}>
            <h1
              style={{
                fontSize: "2.25rem",
                marginBottom: "8px",
                fontWeight: "bold",
              }}
            >
              Sarah Chen
            </h1>
            <h2
              style={{
                fontSize: "1.25rem",
                marginBottom: "16px",
                color: "#666",
              }}
            >
              Award-winning Travel Photographer
            </h2>
          </div>

          <div style={{ marginBottom: "20px" }}>
            <h3
              style={{
                fontSize: "1.125rem",
                marginBottom: "8px",
                fontWeight: "bold",
              }}
            >
              What drives your passion for travel photography?
            </h3>
            <p
              style={{
                fontSize: "1rem",
                lineHeight: 1.5,
                marginBottom: "16px",
                fontStyle: "italic",
              }}
            >
              "Every destination has its own unique rhythm and soul. My camera
              is just a tool to capture those fleeting moments that tell the
              deeper story of a place and its people."
            </p>
          </div>

          <div style={{ marginBottom: "20px" }}>
            <h3
              style={{
                fontSize: "1.125rem",
                marginBottom: "8px",
                fontWeight: "bold",
              }}
            >
              Your most memorable shoot?
            </h3>
            <p
              style={{
                fontSize: "1rem",
                lineHeight: 1.5,
                marginBottom: "16px",
                fontStyle: "italic",
              }}
            >
              "Documenting the sunrise ceremony at Angkor Wat. The interplay of
              ancient architecture, golden light, and spiritual devotion created
              magic that no amount of planning could have orchestrated."
            </p>
          </div>

          <div>
            <h3
              style={{
                fontSize: "1.125rem",
                marginBottom: "8px",
                fontWeight: "bold",
              }}
            >
              Advice for aspiring travel photographers?
            </h3>
            <p
              style={{ fontSize: "1rem", lineHeight: 1.5, fontStyle: "italic" }}
            >
              "Learn to see beyond the obvious. The most powerful images often
              come from quiet observations rather than grand gestures."
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MagazineInterview01;
