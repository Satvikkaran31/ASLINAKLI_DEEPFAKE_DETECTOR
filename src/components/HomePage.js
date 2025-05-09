import React, { useState, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import axios from "axios";
import "./HomePage.css"; 

const HomePage = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  useEffect(() => {
    console.log("Updated Result State:", result);
  }, [result]);

  const { getRootProps, getInputProps } = useDropzone({
    accept: "image/*,video/*",
    onDrop: (acceptedFiles) => {
      setFile(acceptedFiles[0]);
    },
  });

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("image", file);

    try {
      const response = await axios.post("http://127.0.0.1:4000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      console.log("API Response:", response.data);
      setResult(response.data);
    } catch (error) {
      console.error("Error uploading file:", error);
      setResult({ error: "Failed to analyze image" });
    }
    setLoading(false);
  };

  return (
    <div className="homepage-container dark-mode">
      <div className="card">
        <h1 className="main-title">ASLINAKLI</h1>
        <p className="subtitle">Unmasking Deepfakes, One Click at a Time</p>
        <p className="description">Upload an image or paste a URL to check for deepfakes and misinformation.</p>

        <div className="dropzone-container" {...getRootProps()}>
          <input {...getInputProps()} />
          <p className="dropzone-text">{file ? file.name : "Drag & drop or click to select an image"}</p>
        </div>

        <button onClick={handleUpload} className="analyze-button" disabled={!file || loading}>
          {loading ? "Processing..." : "Analyze"}
        </button>

        {result && result.prediction && <p className="result-text fade-in">Result: {result.prediction}</p>}
        {result && result.error && <p className="error-text fade-in">{result.error}</p>}
      </div>
    </div>
  );
};

export default HomePage;
