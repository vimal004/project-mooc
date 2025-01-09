import React, { useState } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState("");

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      alert("Please upload an image first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/predict",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      console.log(response);
      setPrediction(response.data.prediction);
    } catch (error) {
      console.error("Error predicting image:", error);
      alert("Error predicting image. Please try again.");
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h1>Plant Disease Classification</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" accept="image/*" onChange={handleFileChange} />
        <button type="submit">Predict</button>
      </form>
      {prediction && (
        <div style={{ marginTop: "20px" }}>
          <h2>Prediction: {prediction}</h2>
        </div>
      )}
    </div>
  );
}

export default App;
