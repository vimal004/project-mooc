import React, { useState } from "react";
import { UploadCloud, AlertCircle, Loader } from "lucide-react";

const App = () => {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [prompt, setPrompt] = useState("");
  const [ai, setAI] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState("");
  const [preview, setPreview] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError("");
      const reader = new FileReader();
      reader.onloadend = () => setPreview(reader.result);
      reader.readAsDataURL(selectedFile);
    }
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
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setPrediction(data.prediction);
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
      const response = await fetch("http://localhost:5000/gemini", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ prompt: `${prediction} + ${prompt}` }),
      });
      const data = await response.json();
      setAI(data.response);
    } catch (error) {
      console.error("Error generating content:", error);
      setError("Error generating content. Please try again.");
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-green-100 to-green-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-extrabold text-green-900 mb-4">
            Plant Disease Detector 🌿
          </h1>
          <p className="text-lg text-gray-700">
            Upload a plant image to detect diseases and explore AI-driven
            solutions.
          </p>
        </div>

        {error && (
          <div className="mb-6 animate-pulse">
            <div className="bg-red-100 border-l-4 border-red-500 p-4 rounded-lg shadow-md">
              <div className="flex items-center">
                <AlertCircle className="h-6 w-6 text-red-500 mr-2" />
                <p className="text-red-700">{error}</p>
              </div>
            </div>
          </div>
        )}

        <div className="bg-white rounded-3xl shadow-xl p-8 mb-8">
          <form onSubmit={handleSubmit} className="space-y-8">
            <div className="relative">
              <div className="border-4 border-dashed border-gray-300 rounded-xl p-10 text-center hover:border-green-400 transition duration-300">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                />
                <div className="space-y-4">
                  <UploadCloud className="mx-auto h-16 w-16 text-gray-400" />
                  <div className="text-gray-600">
                    {preview ? (
                      <img
                        src={preview}
                        alt="Preview"
                        className="mt-4 mx-auto max-h-48 rounded shadow-md"
                      />
                    ) : (
                      <p className="text-gray-500">
                        Drag & Drop or Click to Upload
                      </p>
                    )}
                  </div>
                </div>
              </div>
            </div>

            <button
              type="submit"
              disabled={isLoading || !file}
              className="w-full flex justify-center items-center py-4 px-6 text-lg font-semibold text-white bg-green-600 hover:bg-green-700 rounded-xl shadow-lg transition duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <Loader className="animate-spin h-6 w-6" />
              ) : (
                "Analyze Image"
              )}
            </button>
          </form>
        </div>

        {prediction && (
          <div className="bg-white rounded-3xl shadow-xl p-8 mb-8 animate-fade-in">
            <h2 className="text-3xl font-bold text-green-900 mb-4">
              Prediction Result 🎯
            </h2>
            <p className="text-lg text-green-700 font-medium">{prediction}</p>
          </div>
        )}

        <div className="bg-white rounded-3xl shadow-xl p-8">
          <form onSubmit={gemini} className="space-y-8">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Ask AI for insights based on the prediction
              </label>
              <input
                type="text"
                placeholder="E.g., Suggested treatments?"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                className="w-full px-5 py-4 rounded-xl border border-gray-300 focus:ring-2 focus:ring-green-500 focus:border-transparent transition duration-300"
              />
            </div>

            <button
              type="submit"
              disabled={isGenerating || !prediction}
              className="w-full flex justify-center items-center py-4 px-6 text-lg font-semibold text-white bg-green-600 hover:bg-green-700 rounded-xl shadow-lg transition duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isGenerating ? (
                <Loader className="animate-spin h-6 w-6" />
              ) : (
                "Get AI Insights"
              )}
            </button>
          </form>

          {ai && (
            <div className="mt-6 p-6 bg-green-50 rounded-xl shadow-md animate-fade-in">
              <h3 className="text-xl font-semibold text-green-900 mb-2">
                AI Insights 🤖
              </h3>
              <p className="text-gray-800 whitespace-pre-wrap">{ai}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;
