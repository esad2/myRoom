// frontend/src/app/page.tsx
"use client";

import { useState, FormEvent } from "react";

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
}

interface ImageAnalysisResponse {
    analysis: ImageAnalysis;
    annotatedImage: string;
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


// --- Main Component ---
export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [analysisType, setAnalysisType] = useState<AnalysisType>("image");
  
  // Using our specific types instead of 'any' for better code quality
  const [imageResult, setImageResult] = useState<ImageAnalysisResponse | null>(null);
  const [videoResult, setVideoResult] = useState<VideoAnalysisResult | null>(null);
  
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [safetyMode, setSafetyMode] = useState<"general" | "child_safety">("general");
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
  if (e.target.files && e.target.files.length > 0) {
    setFile(e.target.files[0]);
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
    const formData = new FormData();
    formData.append("file", file);
    const endpoint = analysisType === 'image' ? "http://localhost:8000/analyze-image/" : "http://localhost:8000/analyze-video/";
    if (analysisType === 'image') {
      formData.append("safety_mode", safetyMode);
    }
    
    try {
      const response = await fetch(endpoint, { method: "POST", body: formData });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "An API error occurred.");
      }
      const data = await response.json();
      if (analysisType === 'image') {
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
  
  const activeTabStyle = "border-blue-500 text-blue-600";
  const inactiveTabStyle = "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300";

  // Helper function to get color styling for the rating badge
  const getRatingColor = (rating: string) => {
    switch (rating?.toLowerCase()) {
      case 'high': return 'bg-red-100 text-red-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'low': return 'bg-blue-100 text-blue-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center p-4 sm:p-8 md:p-12 bg-gray-100">
      <div className="w-full max-w-7xl">
        <header className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-gray-800 tracking-tight">AI Safety Inspector</h1>
          <p className="text-gray-600 mt-2">Upload an image or video for an AI-powered safety analysis.</p>
        </header>

        <div className="w-full bg-white p-6 sm:p-8 rounded-xl shadow-lg">
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8" aria-label="Tabs">
              {(['image', 'video'] as const).map((tab) => (
                <button
                  key={tab}
                  onClick={() => { setAnalysisType(tab); setFile(null); setImageResult(null); setVideoResult(null); }}
                  className={`capitalize whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm ${analysisType === tab ? activeTabStyle : inactiveTabStyle}`}
                >
                  {tab} Analysis
                </button>
              ))}
            </nav>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6 mt-6">
            <div>
              <label htmlFor="file-upload" className="block text-sm font-medium text-gray-700 mb-2">
                Upload {analysisType === 'image' ? 'Image' : 'Video'} File
              </label>
              <input 
                id="file-upload" 
                type="file" 
                key={analysisType} 
                accept={analysisType === 'image' ? "image/*" : "video/*"} 
                onChange={handleFileChange}
                className="mt-1 block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 cursor-pointer"
              />
            </div>

            {analysisType === 'image' && (
              <fieldset>
                <legend className="block text-sm font-medium text-gray-700">Analysis Mode</legend>
                <div className="mt-2 flex gap-x-6 gap-y-2 flex-wrap">
                  {(['general', 'child_safety'] as const).map(mode => (
                    <label key={mode} className="flex items-center cursor-pointer">
                      <input 
                        type="radio" 
                        value={mode} 
                        checked={safetyMode === mode} 
                        onChange={(e) => setSafetyMode(e.target.value as typeof safetyMode)} 
                        className="h-4 w-4 text-blue-600 border-gray-300 focus:ring-blue-500"
                      />
                      <span className="ml-2 block text-sm text-gray-900 capitalize">{mode.replace('_', ' ')}</span>
                    </label>
                  ))}
                </div>
              </fieldset>
            )}

            <button 
              type="submit" 
              disabled={isLoading || !file} 
              className="w-full flex justify-center py-3 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              {isLoading ? `Analyzing ${analysisType}...` : `Analyze ${analysisType.charAt(0).toUpperCase() + analysisType.slice(1)}`}
            </button>
          </form>
        </div>

        {isLoading && <div className="mt-6 text-center text-gray-600 animate-pulse">Analyzing... Please be patient, this may take a moment.</div>}
        {error && <div className="mt-6 text-center text-red-600 bg-red-100 p-4 rounded-lg">{error}</div>}

        {/* --- NEW, ENHANCED Image Results Display --- */}
        {imageResult && analysisType === 'image' && (
          <div className="mt-8 bg-white rounded-xl shadow-lg p-6 sm:p-8">
            <h2 className="text-2xl font-semibold text-gray-900 border-b pb-3 mb-6">Image Analysis Report</h2>
            <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
              
              <div className="lg:col-span-3">
                <h3 className="text-xl font-semibold text-gray-800 mb-4">Annotated Image</h3>
                <img src={`data:image/png;base64,${imageResult.annotatedImage}`} alt="Annotated safety analysis" className="rounded-lg shadow-md w-full border" />
              </div>

              <div className="lg:col-span-2 space-y-6">
                <div>
                  <h3 className="text-xl font-semibold text-gray-800 mb-4">Overall Assessment</h3>
                  <div className="bg-gray-50 p-4 rounded-lg flex justify-around text-center">
                    <div>
                      <p className="text-sm text-gray-500 font-medium">Rating</p>
                      <p className="text-2xl font-bold text-blue-600">{imageResult.analysis.overallSafetyRating}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500 font-medium">Score</p>
                      <p className="text-2xl font-bold text-blue-600">{imageResult.analysis.overallSafetyScore}/100</p>
                    </div>
                  </div>
                </div>
                <div>
                    <h3 className="text-xl font-semibold text-gray-800 mb-4">Identified Items</h3>
                    <div className="space-y-4 max-h-[450px] overflow-y-auto pr-2">
                        {imageResult.analysis.identifiedItems.length > 0 ? (
                            imageResult.analysis.identifiedItems.map((item, index) => (
                                <div key={index} className={`p-4 rounded-lg border-l-4 ${item.isHazard ? 'bg-red-50 border-red-500' : 'bg-green-50 border-green-500'}`}>
                                    <div className="flex justify-between items-start">
                                        <h4 className="font-bold text-gray-800">{item.item}</h4>
                                        <span className={`px-2 py-1 text-xs font-medium rounded-full ${getRatingColor(item.safetyRating)}`}>
                                            {item.safetyRating}
                                        </span>
                                    </div>
                                    {item.isHazard ? (
                                        <>
                                            <p className="text-sm text-red-700 mt-2"><span className="font-semibold">Hazard:</span> {item.hazardDescription}</p>
                                            <p className="text-sm text-gray-800 mt-3 bg-gray-100 p-2 rounded"><span className="font-semibold">Suggestion:</span> {item.resolutionSuggestion}</p>
                                        </>
                                    ) : (
                                        <p className="text-sm text-green-700 mt-1">No significant hazards detected for this item.</p>
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
                 <div>
                  <h3 className="text-xl font-semibold text-gray-800 mb-4">Fun Facts</h3>
                  <div className="text-sm space-y-2 text-gray-700 bg-gray-50 p-4 rounded-lg">
                    <p><strong>Safest Spot:</strong> {imageResult.analysis.funIdeas.safestPlace}</p>
                    <p><strong>Earthquake Spot:</strong> {imageResult.analysis.funIdeas.earthquakeSpot}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Video results display remains unchanged */}
        {videoResult && analysisType === 'video' && (
             <div className="mt-8 bg-white rounded-xl shadow-lg p-6 sm:p-8">
                <h2 className="text-2xl font-semibold text-gray-900 border-b pb-2">Video Analysis Report</h2>
                <div className="bg-blue-50 p-4 rounded-lg my-4">
                  <h3 className="text-lg font-medium text-blue-800">Overall Room Safety Score: <span className="font-bold">{videoResult.overall_room_safety_score}%</span></h3>
                </div>
                <h3 className="text-xl font-semibold text-gray-800 mt-6">Detected Hazard Frames</h3>
                {videoResult.detected_hazards.length > 0 ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mt-4">
                    {videoResult.detected_hazards.map((hazard) => (
                      <div key={hazard.frame} className="border rounded-lg overflow-hidden shadow-sm">
                        <img src={`data:image/jpeg;base64,${hazard.hazard_frame_base64}`} alt={`Hazard at frame ${hazard.frame}`} className="w-full h-auto" />
                        <div className="p-4 bg-red-50">
                          <p className="font-bold text-red-800">Frame {hazard.frame}: {hazard.description}</p>
                          <p className="text-sm text-red-700 mt-1"><span className="font-semibold">Severity:</span> {hazard.severity}</p>
                          <p className="text-sm text-red-700 mt-1"><span className="font-semibold">Suggestion:</span> {hazard.resolution_suggestion}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="mt-4 text-gray-600">No specific hazards were identified in the video.</p>
                )}
              </div>
        )}
      </div>
    </main>
  );
}