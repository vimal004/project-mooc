import React, { useState, useEffect } from "react";
import {
  UploadCloud,
  AlertCircle,
  Loader,
  Camera,
  Maximize2,
  Download,
  Share2,
  Database,
} from "lucide-react";

// TypeWriter component for ChatGPT-like effect
const TypeWriter = ({ text }) => {
  const [displayedText, setDisplayedText] = useState("");
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (!text) return;

    if (currentIndex < text.length) {
      const timer = setTimeout(() => {
        setDisplayedText((prev) => prev + text[currentIndex]);
        setCurrentIndex((prev) => prev + 1);
      }, 20);
      return () => clearTimeout(timer);
    }
  }, [currentIndex, text]);

  const formatText = (content) => {
    if (!content) return null;

    return content.split("\n").map((line, idx) => {
      // Match headings with double asterisks
      const headingMatch = line.match(/^\*\*(.*?):\*\*/);
      if (headingMatch) {
        const heading = headingMatch[1];
        const rest = line.slice(headingMatch[0].length);
        return (
          <div key={idx} className="mb-4">
            <span className="text-xl font-bold text-green-800">{heading}:</span>
            <span className="text-gray-700">{rest}</span>
          </div>
        );
      }
      // Remove any remaining double asterisks in regular text
      const cleanedLine = line.replace(/\*\*(.*?)\*\*/g, "$1");
      return (
        <div key={idx} className="mb-3 text-gray-700">
          {cleanedLine}
        </div>
      );
    });
  };

  return <div className="prose prose-lg">{formatText(displayedText)}</div>;
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
  const [showFullscreen, setShowFullscreen] = useState(false);
  const [history, setHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);

  // Handle file drop
  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith("image/")) {
      handleFileSelection(droppedFile);
    }
  };

  const handleFileSelection = (selectedFile) => {
    setFile(selectedFile);
    setError("");
    const reader = new FileReader();
    reader.onloadend = () => setPreview(reader.result);
    reader.readAsDataURL(selectedFile);
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      handleFileSelection(selectedFile);
    }
  };

  const captureImage = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      const video = document.createElement("video");
      video.srcObject = stream;
      await video.play();

      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext("2d").drawImage(video, 0, 0);

      canvas.toBlob((blob) => {
        handleFileSelection(blob);
        stream.getTracks().forEach((track) => track.stop());
      }, "image/jpeg");
    } catch (err) {
      setError("Camera access failed. Please try uploading an image instead.");
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

      // Add to history
      setHistory((prev) => [
        ...prev,
        {
          image: preview,
          prediction: data.prediction,
          timestamp: new Date().toLocaleString(),
        },
      ]);
    } catch (error) {
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
    setAI(""); // Reset AI response for new animation

    try {
      const response = await fetch("http://localhost:5000/gemini", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: `${prediction} + ${prompt}` }),
      });
      const data = await response.json();
      setAI(data.response);
    } catch (error) {
      setError("Error generating content. Please try again.");
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-emerald-50 to-teal-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-6xl font-black text-green-900 mb-4 bg-gradient-to-r from-green-600 to-teal-600 bg-clip-text text-transparent">
            Plant Disease Detector 🌿
          </h1>
          <p className="text-xl text-gray-700">
            Harness AI to diagnose and treat plant diseases instantly
          </p>
        </div>

        {error && (
          <div className="mb-6 animate-bounce">
            <div className="bg-red-100 border-l-4 border-red-500 p-4 rounded-lg shadow-lg">
              <div className="flex items-center">
                <AlertCircle className="h-6 w-6 text-red-500 mr-2" />
                <p className="text-red-700">{error}</p>
              </div>
            </div>
          </div>
        )}

        <div className="bg-white rounded-3xl shadow-2xl p-8 mb-8 backdrop-blur-lg bg-opacity-90">
          <form onSubmit={handleSubmit} className="space-y-8">
            <div
              className="relative"
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
            >
              <div className="border-4 border-dashed border-green-200 rounded-2xl p-10 text-center hover:border-green-400 transition-all duration-300 transform hover:scale-[1.01]">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                />
                <div className="space-y-4">
                  {preview ? (
                    <div className="relative group">
                      <img
                        src={preview}
                        alt="Preview"
                        className="mx-auto max-h-64 rounded-xl shadow-lg cursor-pointer transition-transform duration-300 group-hover:scale-[1.02]"
                        onClick={() => setShowFullscreen(true)}
                      />
                      <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                        <button
                          type="button"
                          onClick={() => setShowFullscreen(true)}
                          className="p-2 bg-black bg-opacity-50 rounded-full text-white hover:bg-opacity-70"
                        >
                          <Maximize2 className="h-5 w-5" />
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <UploadCloud className="mx-auto h-16 w-16 text-green-400" />
                      <p className="text-gray-500 text-lg">
                        Drop your image here or click to upload
                      </p>
                      <div className="flex justify-center space-x-4">
                        <button
                          type="button"
                          onClick={captureImage}
                          className="flex items-center px-4 py-2 bg-green-100 text-green-700 rounded-lg hover:bg-green-200 transition-colors duration-300"
                        >
                          <Camera className="h-5 w-5 mr-2" />
                          Use Camera
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="flex space-x-4">
              <button
                type="submit"
                disabled={isLoading || !file}
                className="flex-1 flex justify-center items-center py-4 px-6 text-lg font-semibold text-white bg-gradient-to-r from-green-600 to-teal-600 rounded-xl shadow-lg transition-all duration-300 hover:shadow-xl hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
              >
                {isLoading ? (
                  <Loader className="animate-spin h-6 w-6" />
                ) : (
                  "Analyze Image"
                )}
              </button>
              <button
                type="button"
                onClick={() => setShowHistory(!showHistory)}
                className="p-4 bg-gray-100 text-gray-700 rounded-xl hover:bg-gray-200 transition-colors duration-300"
              >
                <Database className="h-6 w-6" />
              </button>
            </div>
          </form>
        </div>

        {prediction && (
          <div className="bg-white rounded-3xl shadow-2xl p-8 mb-8 animate-fade-in backdrop-blur-lg bg-opacity-90">
            <h2 className="text-3xl font-bold text-green-900 mb-4 flex items-center">
              Prediction Result 🎯
              <div className="ml-auto space-x-2">
                <button
                  onClick={() => {
                    /* Add download functionality */
                  }}
                  className="p-2 text-gray-600 hover:text-green-600 transition-colors duration-300"
                >
                  <Download className="h-5 w-5" />
                </button>
                <button
                  onClick={() => {
                    /* Add share functionality */
                  }}
                  className="p-2 text-gray-600 hover:text-green-600 transition-colors duration-300"
                >
                  <Share2 className="h-5 w-5" />
                </button>
              </div>
            </h2>
            <p className="text-xl text-green-700 font-medium">{prediction}</p>
          </div>
        )}

        <div className="bg-white rounded-3xl shadow-2xl p-8 backdrop-blur-lg bg-opacity-90">
          <form onSubmit={gemini} className="space-y-8">
            <div>
              <label className="block text-lg font-medium text-gray-700 mb-2">
                Ask AI for insights
              </label>
              <input
                type="text"
                placeholder="E.g., What are the treatment options? How can I prevent this?"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                className="w-full px-5 py-4 text-lg rounded-xl border border-gray-300 focus:ring-2 focus:ring-green-500 focus:border-transparent transition-all duration-300"
              />
            </div>

            <button
              type="submit"
              disabled={isGenerating || !prediction}
              className="w-full flex justify-center items-center py-4 px-6 text-lg font-semibold text-white bg-gradient-to-r from-green-600 to-teal-600 rounded-xl shadow-lg transition-all duration-300 hover:shadow-xl hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
            >
              {isGenerating ? (
                <Loader className="animate-spin h-6 w-6" />
              ) : (
                "Get AI Insights"
              )}
            </button>
          </form>

          {ai && (
            <div className="mt-8 p-6 bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl shadow-lg animate-fade-in">
              <h3 className="text-2xl font-bold text-green-900 mb-4">
                AI Insights 🤖
              </h3>
              <TypeWriter text={ai} />
            </div>
          )}
        </div>

        {/* History Panel */}
        {showHistory && (
          <div className="fixed inset-y-0 right-0 w-80 bg-white shadow-2xl p-6 transform transition-transform duration-300 overflow-y-auto">
            <h3 className="text-xl font-bold mb-4">History</h3>
            <div className="space-y-4">
              {history.map((item, index) => (
                <div key={index} className="p-4 bg-gray-50 rounded-lg">
                  <img
                    src={item.image}
                    alt="Historical"
                    className="w-full h-32 object-cover rounded mb-2"
                  />
                  <p className="text-sm font-medium">{item.prediction}</p>
                  <p className="text-xs text-gray-500">{item.timestamp}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Fullscreen Image Modal */}
        {showFullscreen && (
          <div
            className="fixed inset-0 bg-black bg-opacity-90 flex items-center justify-center z-50"
            onClick={() => setShowFullscreen(false)}
          >
            <img
              src={preview}
              alt="Fullscreen preview"
              className="max-h-[90vh] max-w-[90vw] object-contain"
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
