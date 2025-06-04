import React from "react";

const MagazineFeature01 = () => {
  return (
    <div style={{ backgroundColor: "white", color: "black", padding: "32px" }}>
      <div style={{ textAlign: "center", marginBottom: "32px" }}>
        <h1
          style={{ fontSize: "3rem", marginBottom: "12px", fontWeight: "bold" }}
        >
          CULINARY ADVENTURES
        </h1>
        <h2
          style={{
            fontSize: "1.5rem",
            marginBottom: "24px",
            fontStyle: "italic",
          }}
        >
          A journey through global flavors
        </h2>
      </div>

      <div style={{ marginBottom: "32px" }}>
        <img
          src="https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=1000"
          alt=""
          style={{ width: "100%", height: "auto", display: "block" }}
        />
      </div>

      <div
        style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "32px" }}
      >
        <div>
          <h3
            style={{
              fontSize: "1.5rem",
              marginBottom: "16px",
              fontWeight: "bold",
            }}
          >
            Street Food Chronicles
          </h3>
          <p
            style={{ fontSize: "1rem", lineHeight: 1.6, marginBottom: "16px" }}
          >
            From Bangkok's bustling night markets to Mexico City's vibrant food
            trucks, street food represents the authentic heart of local cuisine.
            Each bite tells a story of tradition, innovation, and community.
          </p>
          <p style={{ fontSize: "1rem", lineHeight: 1.6 }}>
            Our culinary explorers have traveled to 15 countries, sampling over
            200 different street food dishes to bring you this comprehensive
            guide.
          </p>
        </div>
        <div>
          <h3
            style={{
              fontSize: "1.5rem",
              marginBottom: "16px",
              fontWeight: "bold",
            }}
          >
            Fine Dining Evolution
          </h3>
          <p
            style={{ fontSize: "1rem", lineHeight: 1.6, marginBottom: "16px" }}
          >
            Modern gastronomy is pushing boundaries like never before. Molecular
            gastronomy, plant-based innovations, and sustainable sourcing are
            redefining what it means to dine exceptionally.
          </p>
          <p style={{ fontSize: "1rem", lineHeight: 1.6 }}>
            We explore how renowned chefs are balancing tradition with
            innovation to create unforgettable dining experiences.
          </p>
        </div>
      </div>
    </div>
  );
};

export default MagazineFeature01;
