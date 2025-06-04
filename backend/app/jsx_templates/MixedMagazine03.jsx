import React from "react";

const MixedMagazine03 = () => {
  return (
    <div style={{ backgroundColor: "white", color: "black", padding: "20px" }}>
      <div style={{ display: "flex", gap: "20px", alignItems: "flex-start" }}>
        <div style={{ flex: 1 }}>
          <h1 style={{ fontSize: "2rem", marginBottom: "12px" }}>
            Forest Sanctuary
          </h1>
          <h2 style={{ fontSize: "1.25rem", marginBottom: "16px" }}>
            Where nature finds its voice
          </h2>
          <p
            style={{ fontSize: "1rem", lineHeight: 1.5, marginBottom: "16px" }}
          >
            Deep within the forest, time moves differently. Ancient trees stand
            as silent witnesses to centuries of change, their roots intertwined
            in an underground network of communication and support.
          </p>
          <p
            style={{ fontSize: "1rem", lineHeight: 1.5, marginBottom: "16px" }}
          >
            The forest floor is alive with the rustle of leaves, the scurry of
            small creatures, and the gentle whisper of wind through branches.
          </p>
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
            src="https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400"
            alt=""
            style={{ width: "100%", height: "auto", display: "block" }}
          />
        </div>
      </div>
    </div>
  );
};

export default MixedMagazine03;
