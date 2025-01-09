import React, { useState } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [prompt, setPrompt] = useState("");
  const [ai, setAI] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setError("");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Please upload an image first!");
      return;
    }

    setIsLoading(true);
    setError("");

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
      setPrediction(response.data.prediction);
    } catch (error) {
      console.error("Error predicting image:", error);
      setError("Error predicting image. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const gemini = async (e) => {
    e.preventDefault();
    if (!prediction) {
      setError("Please make a prediction first!");
      return;
    }

    setIsGenerating(true);
    setError("");

    try {
      const response = await axios.post("http://localhost:5000/gemini", {
        prompt: `${prediction} + ${prompt}`,
      });
      setAI(response.data.response);
    } catch (error) {
      console.error("Error generating content:", error);
      setError("Error generating content. Please try again.");
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100 py-12 px-4 sm:px-6 lg:px-8">
      <h1 className="text-3xl font-extrabold text-gray-900 mb-10">
        Plant Disease Classification
      </h1>

      {error && (
        <div
          className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4"
          role="alert"
        >
          <p>{error}</p>
        </div>
      )}

      <form onSubmit={handleSubmit} className="mb-8">
        <div className="flex items-center space-x-4">
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5"
          />
          <button
            type="submit"
            className="bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 transition duration-300"
            disabled={isLoading}
          >
            {isLoading ? (
              <svg
                className="animate-spin h-5 w-5 mr-3 text-white"
                viewBox="0 0 24 24"
              ></svg>
            ) : (
              "Predict"
            )}
          </button>
        </div>
      </form>

      {prediction && (
        <div className="text-xl font-bold text-green-600 mb-8">
          <h2 className="text-2xl font-extrabold mb-2">Prediction:</h2>
          <p className="text-xl">{prediction}</p>
        </div>
      )}

      <form onSubmit={gemini} className="mb-8">
        <div className="flex items-center space-x-4">
          <input
            type="text"
            placeholder="Enter your query"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5"
          />
          <button
            type="submit"
            className="bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 transition duration-300"
            disabled={isGenerating}
          >
            {isGenerating ? (
              <svg
                className="animate-spin h-5 w-5 mr-3 text-white"
                viewBox="0 0 24 24"
              ></svg>
            ) : (
              "Generate Content"
            )}
          </button>
        </div>
      </form>

      <div>
        <h1 className="text-2xl font-extrabold mb-2">AI Response</h1>
        <p className="text-xl">{ai}</p>
      </div>
    </div>
  );
}

export default App;
