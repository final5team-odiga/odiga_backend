import React from "react";

const MixedMagazine05 = () => {
  return (
    <div style={{ backgroundColor: "white", color: "black", padding: "20px" }}>
      <h1 style={{ fontSize: "2.5rem", marginBottom: "8px" }}>Urban Nature</h1>
      <h2 style={{ fontSize: "1.25rem", marginBottom: "20px" }}>
        Finding green spaces in concrete jungles
      </h2>

      <div style={{ marginBottom: "24px" }}>
        <img
          src="https://images.unsplash.com/photo-1439066615861-d1af74d74000?w=800"
          alt=""
          style={{ width: "100%", height: "auto", display: "block" }}
        />
      </div>

      <div style={{ display: "flex", gap: "20px" }}>
        <div style={{ flex: 2 }}>
          <p
            style={{
              fontSize: "1.125rem",
              lineHeight: 1.6,
              marginBottom: "16px",
            }}
          >
            In the heart of bustling cities, pockets of nature provide essential
            respite from urban life. Parks, gardens, and green corridors serve
            as vital lungs for metropolitan areas, offering both environmental
            benefits and psychological relief to city dwellers.
          </p>
          <p style={{ fontSize: "1.125rem", lineHeight: 1.6 }}>
            These urban oases demonstrate that nature and human development can
            coexist harmoniously, creating spaces where both can thrive.
          </p>
        </div>
        <div style={{ flex: 1 }}>
          <img
            src="https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=300"
            alt=""
            style={{ width: "100%", height: "auto", display: "block" }}
          />
        </div>
      </div>
    </div>
  );
};

export default MixedMagazine05;
