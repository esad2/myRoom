// frontend/src/app/page.tsx
"use client";

import { useState, FormEvent, useRef, useEffect } from "react";

// --- Type Definitions for API Responses ---
type AnalysisType = "image" | "video";

// 1. Specific Interfaces for Image Analysis Data
interface IdentifiedItem {
  item: string;
  safetyRating: string; // e.g., "High", "Medium", "Low"
  isHazard: boolean;
  hazardDescription: string;
  resolutionSuggestion: string;
  coordinates?: { x: number; y: number; width: number; height: number };
}

interface FunIdeas {
  safestPlace: string;
  earthquakeSpot: string;
}

interface ImageAnalysis {
  overallSafetyRating: string;
  overallSafetyScore: number;
  identifiedItems: IdentifiedItem[];
  funIdeas: FunIdeas;
  funFactAboutRoom: string; // Added this based on your backend schema
}

interface ImageAnalysisResponse {
  analysis: ImageAnalysis;
  originalImage: string; // CHANGED: now originalImage, not annotatedImage
}

// 2. Interfaces for Video Analysis (Unchanged)
interface VideoHazard {
  frame: number;
  description: string;
  severity: string;
  resolution_suggestion: string;
  hazard_frame_base64: string;
}

interface VideoAnalysisResult {
  status: string;
  overall_room_safety_score: number;
  detected_hazards: VideoHazard[];
}

// --- Image Display Component with Interactive Hazards ---
interface ImageDisplayProps {
  imageData: string; // Base64 string of the original image
  hazards: IdentifiedItem[];
  onHazardClick: (index: number) => void;
  selectedHazardIndex: number | null;
}

const ImageDisplay: React.FC<ImageDisplayProps> = ({
  imageData,
  hazards,
  onHazardClick,
  selectedHazardIndex,
}) => {
  const imgRef = useRef<HTMLImageElement>(null);
  const [naturalDimensions, setNaturalDimensions] = useState({ width: 0, height: 0 });
  const [renderedDimensions, setRenderedDimensions] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const updateDimensions = () => {
      if (imgRef.current) {
        // Get natural (original) dimensions of the image
        setNaturalDimensions({
          width: imgRef.current.naturalWidth,
          height: imgRef.current.naturalHeight,
        });
        // Get rendered (displayed) dimensions of the image
        setRenderedDimensions({
          width: imgRef.current.offsetWidth,
          height: imgRef.current.offsetHeight,
        });
      }
    };

    const imgElement = imgRef.current;
    if (imgElement) {
      // If image is already loaded, update dimensions immediately
      if (imgElement.complete) {
        updateDimensions();
      } else {
        // Otherwise, wait for the image to load
        imgElement.onload = updateDimensions;
      }
    }

    // Add event listener for window resize to handle responsiveness
    window.addEventListener("resize", updateDimensions);

    return () => {
      window.removeEventListener("resize", updateDimensions);
      if (imgElement) {
        imgElement.onload = null; // Clean up onload handler
      }
    };
  }, [imageData]); // Re-run effect if imageData changes

  // Filter for actual hazards with coordinates
  const hazardsWithCoords = hazards.filter(
    (item) => item.isHazard && item.coordinates
  );

  // Calculate scaling factor
  const scaleX = renderedDimensions.width / naturalDimensions.width;
  const scaleY = renderedDimensions.height / naturalDimensions.height;

  // We assume that the image is scaled proportionally (object-contain).
  // If the container's aspect ratio differs from the image's,
  // one of these scales will be the limiting factor, and the other will result in
  // whitespace (letterboxing/pillarboxing).
  // We need to use the *smaller* of the two scales to ensure coordinates map correctly
  // onto the visible portion of the image.

  const actualScale = Math.min(scaleX, scaleY);

  // Calculate the offset for pillarboxing/letterboxing
  const offsetX = (renderedDimensions.width - (naturalDimensions.width * actualScale)) / 2;
  const offsetY = (renderedDimensions.height - (naturalDimensions.height * actualScale)) / 2;


  return (
    <div className="relative w-full h-full max-h-[80vh] flex justify-center items-center overflow-hidden rounded-lg shadow-md bg-gray-200">
      <img
        ref={imgRef}
        src={`data:image/jpeg;base64,${imageData}`}
        alt="Room for safety analysis"
        className="block max-w-full max-h-full object-contain"
        // Ensure image occupies full width/height of its container for measurement,
        // while object-contain ensures aspect ratio is maintained within that space.
        style={{ width: "100%", height: "100%" }}
      />

      {naturalDimensions.width > 0 &&
        naturalDimensions.height > 0 &&
        renderedDimensions.width > 0 &&
        renderedDimensions.height > 0 &&
        hazardsWithCoords.map((hazard, index) => {
          const { x, y, width, height } = hazard.coordinates!; // Assert non-null after filter

          // Calculate pixel positions from normalized coordinates
          // Apply the actual scale and then the offset
          const left = (x * naturalDimensions.width * actualScale) + offsetX;
          const top = (y * naturalDimensions.height * actualScale) + offsetY;

          const isSelected = selectedHazardIndex === index;

          return (
            <button
              key={index}
              className={`absolute flex items-center justify-center
                          w-8 h-8 rounded-full text-white text-sm font-bold cursor-pointer
                          transition-all duration-200 ease-in-out
                          ${
                            isSelected
                              ? "bg-blue-600 ring-4 ring-blue-300 scale-125"
                              : "bg-red-500 hover:bg-red-600"
                          }`}
              style={{
                left: `${left}px`,
                top: `${top}px`,
                // Optional: Adjust if the marker itself has width/height, to center it.
                // transform: 'translate(-50%, -50%)', // Centers the marker on the top-left of the bounding box
              }}
              onClick={() => onHazardClick(index)}
              title={hazard.item}
            >
              {index + 1}
            </button>
          );
        })}
    </div>
  );
};

// --- Main Component ---
export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [analysisType, setAnalysisType] = useState<AnalysisType>("image");

  // Using our specific types instead of 'any' for better code quality
  const [imageResult, setImageResult] =
    useState<ImageAnalysisResponse | null>(null);
  const [videoResult, setVideoResult] =
    useState<VideoAnalysisResult | null>(null);

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [safetyMode, setSafetyMode] = useState<
    "general" | "child_safety" | "custom"
  >("general"); // Added 'custom' to the frontend state
  const [customSafetyText, setCustomSafetyText] = useState<string>(""); // State for custom text

  const [selectedHazardIndex, setSelectedHazardIndex] = useState<number | null>(
    null
  ); // New state for selected hazard

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
      // Clear results when a new file is selected
      setImageResult(null);
      setVideoResult(null);
      setSelectedHazardIndex(null); // Clear selected hazard
    } else {
      setFile(null);
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError("Please select a file first.");
      return;
    }
    setIsLoading(true);
    setError(null);
    setImageResult(null);
    setVideoResult(null);
    setSelectedHazardIndex(null); // Clear selected hazard before new analysis

    const formData = new FormData();
    formData.append("file", file);
    const endpoint =
      analysisType === "image"
        ? "http://localhost:8000/analyze-image/"
        : "http://localhost:8000/analyze-video/";

    if (analysisType === "image") {
      formData.append("safety_mode", safetyMode);
      if (safetyMode === "custom") {
        // Our backend's analyze_single_room_image expects custom_safety_text
        // to be passed directly as an argument, which is then incorporated into the prompt.
        // It does NOT expect it as a separate form field named "custom_safety_text".
        // The current backend implementation handles this implicitly within generate_prompt_for_mode
        // by constructing the full prompt string.
        // So, no need to append it directly to formData for the current backend.
        // If your backend changed to accept it directly via endpoint, you would uncomment:
        // formData.append("custom_safety_text", customSafetyText);
      }
    }

    try {
      const response = await fetch(endpoint, { method: "POST", body: formData });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "An API error occurred.");
      }
      const data = await response.json();
      if (analysisType === "image") {
        setImageResult(data);
      } else {
        setVideoResult(data);
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleHazardClick = (index: number) => {
    setSelectedHazardIndex(index === selectedHazardIndex ? null : index);
  };

  const activeTabStyle = "border-blue-500 text-blue-600";
  const inactiveTabStyle =
    "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300";

  // Helper function to get color styling for the rating badge
  const getRatingColor = (rating: string) => {
    switch (rating?.toLowerCase()) {
      case "high":
        return "bg-red-100 text-red-800";
      case "medium":
        return "bg-yellow-100 text-yellow-800";
      case "low":
        return "bg-blue-100 text-blue-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  // Filter hazards that have coordinates and are actual hazards for rendering markers
  const hazardsToDisplay =
    imageResult?.analysis.identifiedItems.filter(
      (item) => item.isHazard && item.coordinates
    ) || [];

  return (
    <main className="flex flex-col min-h-screen p-4 sm:p-8 md:p-12 bg-gray-100">
      <div className="w-full max-w-7xl mx-auto">
        <header className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-gray-800 tracking-tight">
            AI Safety Inspector
          </h1>
          <p className="text-gray-600 mt-2">
            Upload an image or video for an AI-powered safety analysis.
          </p>
        </header>

        <div className="w-full bg-white p-6 sm:p-8 rounded-xl shadow-lg mb-8">
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8" aria-label="Tabs">
              {([`image`, `video`] as const).map((tab) => (
                <button
                  key={tab}
                  onClick={() => {
                    setAnalysisType(tab);
                    setFile(null);
                    setImageResult(null);
                    setVideoResult(null);
                    setSelectedHazardIndex(null);
                    setError(null); // Clear errors on tab switch
                  }}
                  className={`capitalize whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm ${
                    analysisType === tab ? activeTabStyle : inactiveTabStyle
                  }`}
                >
                  {tab} Analysis
                </button>
              ))}
            </nav>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6 mt-6">
            <div>
              <label
                htmlFor="file-upload"
                className="block text-sm font-medium text-gray-700 mb-2"
              >
                Upload {analysisType === "image" ? "Image" : "Video"} File
              </label>
              <input
                id="file-upload"
                type="file"
                key={analysisType}
                accept={analysisType === "image" ? "image/*" : "video/*"}
                onChange={handleFileChange}
                className="mt-1 block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 cursor-pointer"
              />
            </div>

            {analysisType === "image" && (
              <fieldset>
                <legend className="block text-sm font-medium text-gray-700">
                  Analysis Mode
                </legend>
                <div className="mt-2 flex gap-x-6 gap-y-2 flex-wrap">
                  {([`general`, `child_safety`, `custom`] as const).map(
                    (mode) => (
                      <label key={mode} className="flex items-center cursor-pointer">
                        <input
                          type="radio"
                          value={mode}
                          checked={safetyMode === mode}
                          onChange={(e) => {
                            setSafetyMode(e.target.value as typeof safetyMode);
                            if (e.target.value !== 'custom') {
                              setCustomSafetyText(''); // Clear custom text if mode is not custom
                            }
                          }}
                          className="h-4 w-4 text-blue-600 border-gray-300 focus:ring-blue-500"
                        />
                        <span className="ml-2 block text-sm text-gray-900 capitalize">
                          {mode.replace(`_`, ` `)}
                        </span>
                      </label>
                    )
                  )}
                </div>
                {safetyMode === 'custom' && (
                  <div className="mt-4">
                    <label htmlFor="custom-safety-text" className="block text-sm font-medium text-gray-700 mb-2">
                      Custom Safety Guidelines:
                    </label>
                    <textarea
                      id="custom-safety-text"
                      rows={3}
                      value={customSafetyText}
                      onChange={(e) => setCustomSafetyText(e.target.value)}
                      placeholder="e.g., 'Ensure no sharp tools are exposed.', 'Check for proper ventilation in a workshop.'"
                      className="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm p-2"
                    ></textarea>
                  </div>
                )}
              </fieldset>
            )}

            <button
              type="submit"
              disabled={isLoading || !file}
              className="w-full flex justify-center py-3 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              {isLoading
                ? `Analyzing ${analysisType}...`
                : `Analyze ${
                    analysisType.charAt(0).toUpperCase() + analysisType.slice(1)
                  }`}
            </button>
          </form>
        </div>

        {isLoading && (
          <div className="mt-6 text-center text-gray-600 animate-pulse">
            Analyzing... Please be patient, this may take a moment.
          </div>
        )}
        {error && (
          <div className="mt-6 text-center text-red-600 bg-red-100 p-4 rounded-lg">
            {error}
          </div>
        )}

        {/* --- NEW, ENHANCED Image Results Display --- */}
        {imageResult && analysisType === "image" && (
          <div className="mt-8 bg-white rounded-xl shadow-lg p-6 sm:p-8 flex flex-col lg:flex-row gap-8">
            {/* Image Display Area (takes 60% on large screens) */}
            <div className="lg:w-3/5 flex-shrink-0">
              <h2 className="text-2xl font-semibold text-gray-900 border-b pb-3 mb-6">
                Room Safety Report
              </h2>
              <ImageDisplay
                imageData={imageResult.originalImage}
                hazards={imageResult.analysis.identifiedItems}
                onHazardClick={handleHazardClick}
                selectedHazardIndex={selectedHazardIndex}
              />
            </div>

            {/* Analysis Details (takes 40% on large screens) */}
            <div className="lg:w-2/5 space-y-6 flex flex-col">
              <div>
                <h3 className="text-xl font-semibold text-gray-800 mb-4">
                  Overall Assessment
                </h3>
                <div className="bg-gray-50 p-4 rounded-lg flex justify-around text-center">
                  <div>
                    <p className="text-sm text-gray-500 font-medium">Rating</p>
                    <p className="text-2xl font-bold text-blue-600">
                      {imageResult.analysis.overallSafetyRating}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500 font-medium">Score</p>
                    <p className="text-2xl font-bold text-blue-600">
                      {imageResult.analysis.overallSafetyScore}/100
                    </p>
                  </div>
                </div>
              </div>

              {selectedHazardIndex !== null && hazardsToDisplay[selectedHazardIndex] ? (
                // Display details of the selected hazard
                <div className="mt-4 p-4 rounded-lg bg-red-50 border border-red-300">
                  <h4 className="font-bold text-red-800 text-lg mb-2">
                    Hazard {selectedHazardIndex + 1}: {hazardsToDisplay[selectedHazardIndex].item}
                  </h4>
                  <p className="text-sm text-red-700 mb-2">
                    <span className="font-semibold">Description:</span>{" "}
                    {hazardsToDisplay[selectedHazardIndex].hazardDescription}
                  </p>
                  <p className="text-sm text-gray-800 bg-gray-100 p-2 rounded">
                    <span className="font-semibold">Suggestion:</span>{" "}
                    {hazardsToDisplay[selectedHazardIndex].resolutionSuggestion}
                  </p>
                </div>
              ) : (
                // Display general identified items list if no specific hazard is selected
                <div>
                  <h3 className="text-xl font-semibold text-gray-800 mb-4">
                    Identified Items (Click a number on the image for details)
                  </h3>
                  <div className="space-y-4 max-h-[450px] overflow-y-auto pr-2">
                    {imageResult.analysis.identifiedItems.length > 0 ? (
                      imageResult.analysis.identifiedItems.map((item, index) => (
                        <div
                          key={index}
                          className={`p-4 rounded-lg border-l-4 ${
                            item.isHazard
                              ? "bg-red-50 border-red-500"
                              : "bg-green-50 border-green-500"
                          }`}
                        >
                          <div className="flex justify-between items-start">
                            <h4 className="font-bold text-gray-800">{item.item}</h4>
                            <span
                              className={`px-2 py-1 text-xs font-medium rounded-full ${getRatingColor(
                                item.safetyRating
                              )}`}
                            >
                              {item.safetyRating}
                            </span>
                          </div>
                          {item.isHazard ? (
                            <>
                              <p className="text-sm text-red-700 mt-2">
                                <span className="font-semibold">Hazard:</span>{" "}
                                {item.hazardDescription}
                              </p>
                              {/* Removed resolution suggestion here, shown when clicked */}
                            </>
                          ) : (
                            <p className="text-sm text-green-700 mt-1">
                              No significant hazards detected for this item.
                            </p>
                          )}
                        </div>
                      ))
                    ) : (
                      <div className="text-center py-10 text-gray-500">
                        <p>No items were identified in the analysis.</p>
                      </div>
                    )}
                  </div>
                </div>
              )}

              <div>
                <h3 className="text-xl font-semibold text-gray-800 mb-4">
                  Fun Facts
                </h3>
                <div className="text-sm space-y-2 text-gray-700 bg-gray-50 p-4 rounded-lg">
                  <p>
                    <strong>Safest Spot:</strong>{" "}
                    {imageResult.analysis.funIdeas.safestPlace}
                  </p>
                  <p>
                    <strong>Earthquake Spot:</strong>{" "}
                    {imageResult.analysis.funIdeas.earthquakeSpot}
                  </p>
                  <p>
                    <strong>Fun Fact about Room:</strong>{" "}
                    {imageResult.analysis.funFactAboutRoom}
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Video results display remains unchanged */}
        {videoResult && analysisType === "video" && (
          <div className="mt-8 bg-white rounded-xl shadow-lg p-6 sm:p-8">
            <h2 className="text-2xl font-semibold text-gray-900 border-b pb-2">
              Video Analysis Report
            </h2>
            <div className="bg-blue-50 p-4 rounded-lg my-4">
              <h3 className="text-lg font-medium text-blue-800">
                Overall Room Safety Score:{" "}
                <span className="font-bold">
                  {videoResult.overall_room_safety_score}%
                </span>
              </h3>
            </div>
            <h3 className="text-xl font-semibold text-gray-800 mt-6">
              Detected Hazard Frames
            </h3>
            {videoResult.detected_hazards.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mt-4">
                {videoResult.detected_hazards.map((hazard) => (
                  <div
                    key={hazard.frame}
                    className="border rounded-lg overflow-hidden shadow-sm"
                  >
                    <img
                      src={`data:image/jpeg;base64,${hazard.hazard_frame_base64}`}
                      alt={`Hazard at frame ${hazard.frame}`}
                      className="w-full h-auto"
                    />
                    <div className="p-4 bg-red-50">
                      <p className="font-bold text-red-800">
                        Frame {hazard.frame}: {hazard.description}
                      </p>
                      <p className="text-sm text-red-700 mt-1">
                        <span className="font-semibold">Severity:</span>{" "}
                        {hazard.severity}
                      </p>
                      <p className="text-sm text-red-700 mt-1">
                        <span className="font-semibold">Suggestion:</span>{" "}
                        {hazard.resolution_suggestion}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="mt-4 text-gray-600">
                No specific hazards were identified in the video.
              </p>
            )}
          </div>
        )}
      </div>
    </main>
  );
}