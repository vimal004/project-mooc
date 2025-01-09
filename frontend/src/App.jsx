import React, { useState, useEffect } from "react";
import { UploadCloud, AlertCircle, Loader } from "lucide-react";

// TypeWriter component for smooth text animation
const TypeWriter = ({ text }) => {
  const [displayedText, setDisplayedText] = useState("");
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (currentIndex < text.length) {
      const timer = setTimeout(() => {
        setDisplayedText((prev) => prev + text[currentIndex]);
        setCurrentIndex((prev) => prev + 1);
      }, 20); // Adjust speed here
      return () => clearTimeout(timer);
    }
  }, [currentIndex, text]);

  // Format text to handle markdown-style formatting
  const formatText = (text) => {
    const lines = text.split("\n");
    return lines.map((line, index) => {
      // Handle bold text for headings (e.g., "**Treatment:**")
      const headingMatch = line.match(/^\*\*(.*?)\*\*/);
      if (headingMatch) {
        const heading = headingMatch[1];
        const rest = line.slice(headingMatch[0].length);
        return (
          <div key={index} className="mb-2">
            <span className="font-bold text-gray-900">{heading}</span>
            <span>{rest}</span>
          </div>
        );
      }
      // Remove unnecessary double asterisks within text
      const cleanedLine = line.replace(/\*\*(.*?)\*\*/g, "$1");
      return (
        <div key={index} className="mb-2">
          {cleanedLine}
        </div>
      );
    });
  };

  return <div className="prose">{formatText(displayedText)}</div>;
};

const App = () => {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [prompt, setPrompt] = useState("");
  const [ai, setAI] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState("");
  const [preview, setPreview] = useState(null);
  const [showResponse, setShowResponse] = useState(false);

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
    setShowResponse(false);

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
      setShowResponse(true);
    } catch (error) {
      console.error("Error generating content:", error);
      setError("Error generating content. Please try again.");
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Plant Disease Classification
          </h1>
          <p className="text-gray-600">
            Upload a plant image to detect diseases and get AI-powered insights
          </p>
        </div>

        {error && (
          <div className="mb-6 animate-fade-in">
            <div className="bg-red-50 border-l-4 border-red-400 p-4 rounded-r">
              <div className="flex items-center">
                <AlertCircle className="h-5 w-5 text-red-400 mr-2" />
                <p className="text-red-700">{error}</p>
              </div>
            </div>
          </div>
        )}

        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="relative">
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-gray-400 transition-colors duration-200">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                />
                <div className="space-y-2">
                  <UploadCloud className="mx-auto h-12 w-12 text-gray-400" />
                  <div className="text-gray-600">
                    {preview ? (
                      <img
                        src={preview}
                        alt="Preview"
                        className="mt-4 mx-auto max-h-48 rounded"
                      />
                    ) : (
                      <p>Drop your image here or click to browse</p>
                    )}
                  </div>
                </div>
              </div>
            </div>

            <button
              type="submit"
              disabled={isLoading || !file}
              className="w-full flex justify-center items-center py-3 px-4 border border-transparent rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
            >
              {isLoading ? (
                <Loader className="animate-spin h-5 w-5" />
              ) : (
                "Analyze Image"
              )}
            </button>
          </form>
        </div>

        {prediction && (
          <div className="bg-white rounded-xl shadow-lg p-6 mb-8 animate-fade-in">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">
              Prediction Result
            </h2>
            <p className="text-lg text-green-600 font-medium">{prediction}</p>
          </div>
        )}

        <div className="bg-white rounded-xl shadow-lg p-6">
          <form onSubmit={gemini} className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Ask AI about the prediction
              </label>
              <input
                type="text"
                placeholder="E.g., What are the treatment options?"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                className="w-full px-4 py-3 rounded-md border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
              />
            </div>

            <button
              type="submit"
              disabled={isGenerating || !prediction}
              className="w-full flex justify-center items-center py-3 px-4 border border-transparent rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
            >
              {isGenerating ? (
                <Loader className="animate-spin h-5 w-5" />
              ) : (
                "Get AI Insights"
              )}
            </button>
          </form>

          {showResponse && ai && (
            <div className="mt-6 p-4 bg-gray-50 rounded-lg animate-fade-in">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                AI Response
              </h3>
              <div className="text-gray-700">
                <TypeWriter text={ai} />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;
