import { useState, useRef, ChangeEvent } from "react";
import "./App.css";
import axios from "axios";

function App() {
  interface Data {
    confidence_scores: {
      angry: number;
      fearful: number;
      neutral: number;
      surprise: number;
    };
    predicted_emotion: string;
  }

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [emotionData, setEmotionData] = useState<Data | null>(null);

  const handleSubmit = async (e: ChangeEvent<HTMLFormElement>) => {
    e.preventDefault();

    if (!selectedFile) {
      alert("Please select an audio file.");
      return;
    }

    const formData = new FormData();
    formData.append("audio", selectedFile);
    const data = await axios.post(
      "http://127.0.0.1:5000/predict_emotion",
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      }
    );
    console.log(data);
    setEmotionData(data.data);
  };

  const getEmoji = (emotion: string) => {
    switch (emotion) {
      case "angry":
        return "😠";
      case "fearful":
        return "😨";
      case "neutral":
        return "😐";
      case "surprise":
        return "😲";
      default:
        return "😶";
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
      <form onSubmit={handleSubmit} className="flex flex-col items-center">
        <input
          type="file"
          ref={fileInputRef}
          onChange={(e) => {
            if (e.target.files && e.target.files.length > 0) {
              setSelectedFile(e.target.files[0]);
            }
          }}
          className="hidden"
        />
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="border border-gray-700 px-4 py-2 text-gray-700 hover:bg-gray-700 hover:text-white focus:outline-none focus:bg-gray-700"
        >
          Select a File
        </button>
        {selectedFile && (
          <div className="mt-2">
            <p className="text-gray-700">Selected File:</p>
            <p className="text-gray-800">{selectedFile.name}</p>
          </div>
        )}
        <button
          type="submit"
          className="mt-4 px-6 py-2 bg-blue-600 text-white hover:bg-blue-700 focus:outline-none"
        >
          Check Emotion
        </button>
      </form>

      {emotionData && (
        <div className="mt-10 max-w-sm p-6 bg-white shadow-md rounded-lg flex flex-col items-center">
          <h2 className="text-lg font-semibold text-gray-700 mb-4">
            Predicted Emotion
          </h2>
          <div className="text-center">
            <span className="text-4xl">
              {getEmoji(emotionData.predicted_emotion)}
            </span>
            <p className="text-xl font-medium text-gray-800 mt-2">
              {emotionData.predicted_emotion.charAt(0).toUpperCase() +
                emotionData.predicted_emotion.slice(1)}
            </p>
          </div>

          <h3 className="mt-6 text-md font-semibold text-gray-600">
            Confidence Scores
          </h3>
          <ul className="mt-2 space-y-2 text-center">
            {Object.entries(emotionData.confidence_scores).map(
              ([emotion, score]) => (
                <li key={emotion} className="text-gray-700">
                  <span className="capitalize font-semibold">{emotion}</span>:{" "}
                  {score > 0 ? (score * 100).toFixed(2) + "%" : "0%"}
                </li>
              )
            )}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
