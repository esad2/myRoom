// frontend/src/app/page.tsx
"use client"; // This is a client component, as it uses hooks and event handlers

import { useState, FormEvent } from "react";

// Define TypeScript types for the API response to ensure type safety
interface Coordinates {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface IdentifiedItem {
  item: string;
  safetyRating: string;
  isHazard: boolean;
  hazardDescription: string;
  resolutionSuggestion: string;
  coordinates?: Coordinates;
}

interface FunIdeas {
  safestPlace: string;
  earthquakeSpot: string;
}

interface AnalysisResult {
  overallSafetyRating: string;
  overallSafetyScore: number;
  identifiedItems: IdentifiedItem[];
  funIdeas: FunIdeas;
}

interface ApiResponse {
  analysis: AnalysisResult;
  annotatedImage: string | null; // Base64 encoded image string
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [safetyMode, setSafetyMode] = useState<"general" | "child_safety">("general");
  const [result, setResult] = useState<ApiResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError("Please select an image file first.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("safety_mode", safetyMode);

    try {
      const response = await fetch("http://localhost:8000/analyze/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "An unknown error occurred.");
      }

      const data: ApiResponse = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center p-8 sm:p-12 md:p-24 bg-gray-50">
      <div className="z-10 w-full max-w-5xl items-center justify-between font-mono text-sm lg:flex mb-12">
        <h1 className="text-3xl font-bold text-gray-800">AI Safety Inspector</h1>
      </div>

      <div className="w-full max-w-5xl bg-white p-8 rounded-lg shadow-md">
        {/* Form Section */}
        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label htmlFor="file-upload" className="block text-sm font-medium text-gray-700">
              Upload Room Image
            </label>
            <input
              id="file-upload"
              type="file"
              accept="image/png, image/jpeg, image/gif"
              onChange={handleFileChange}
              className="mt-1 block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
            />
          </div>

          <fieldset>
            <legend className="block text-sm font-medium text-gray-700">Analysis Mode</legend>
            <div className="mt-2 space-y-2">
              <div className="flex items-center">
                <input id="general" name="safety-mode" type="radio" value="general" checked={safetyMode === 'general'} onChange={() => setSafetyMode('general')} className="h-4 w-4 text-blue-600 border-gray-300 focus:ring-blue-500" />
                <label htmlFor="general" className="ml-3 block text-sm font-medium text-gray-700">General Safety</label>
              </div>
              <div className="flex items-center">
                <input id="child_safety" name="safety-mode" type="radio" value="child_safety" checked={safetyMode === 'child_safety'} onChange={() => setSafetyMode('child_safety')} className="h-4 w-4 text-blue-600 border-gray-300 focus:ring-blue-500" />
                <label htmlFor="child_safety" className="ml-3 block text-sm font-medium text-gray-700">Child Safety</label>
              </div>
            </div>
          </fieldset>
          
          <button
            type="submit"
            disabled={isLoading || !file}
            className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            {isLoading ? "Analyzing..." : "Analyze Room"}
          </button>
        </form>

        {/* Loading and Error State */}
        {isLoading && <div className="mt-6 text-center">Loading results, this may take a moment...</div>}
        {error && <div className="mt-6 text-center text-red-600 bg-red-100 p-3 rounded-md">{error}</div>}

        {/* Results Section */}
        {result && (
          <div className="mt-10">
            <h2 className="text-2xl font-semibold text-gray-900 border-b pb-2">Analysis Report</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-6">
                {/* Annotated Image */}
                {result.annotatedImage ? (
                     <div>
                        <h3 className="text-lg font-medium text-gray-800">Highlighted Hazards</h3>
                        <img src={`data:image/jpeg;base64,${result.annotatedImage}`} alt="Annotated room" className="mt-2 rounded-lg shadow-sm w-full" />
                    </div>
                ) : <div className="text-gray-600">No hazards found to highlight.</div>}


                {/* Safety Score */}
                <div className="bg-gray-50 p-4 rounded-lg">
                    <h3 className="text-lg font-medium text-gray-800">Overall Score: <span className="font-bold text-blue-600">{result.analysis.overallSafetyScore}/100</span></h3>
                    <p className="text-md text-gray-600">Rating: <span className="font-semibold">{result.analysis.overallSafetyRating}</span></p>
                </div>
            </div>

            {/* Identified Items */}
            <div className="mt-8">
              <h3 className="text-lg font-medium text-gray-800">Identified Items</h3>
              <ul className="mt-4 space-y-4">
                {result.analysis.identifiedItems.map((item, index) => (
                  <li key={index} className={`p-4 rounded-lg ${item.isHazard ? 'bg-red-50 border border-red-200' : 'bg-green-50 border border-green-200'}`}>
                    <p className="font-semibold text-gray-800">{item.item} - <span className={item.isHazard ? 'text-red-700' : 'text-green-700'}>{item.isHazard ? "HAZARD" : "SAFE"}</span></p>
                    {item.isHazard && (
                        <div className="mt-2 text-sm text-gray-600 space-y-1">
                            <p><span className="font-medium">Description:</span> {item.hazardDescription}</p>
                            <p><span className="font-medium">Suggestion:</span> {item.resolutionSuggestion}</p>
                        </div>
                    )}
                  </li>
                ))}
              </ul>
            </div>
             {/* Fun Ideas */}
            <div className="mt-8 bg-indigo-50 p-4 rounded-lg">
                <h3 className="text-lg font-medium text-indigo-800">Fun Ideas!</h3>
                <div className="mt-2 text-sm text-indigo-700 space-y-1">
                    <p><strong>Safest Spot:</strong> {result.analysis.funIdeas.safestPlace}</p>
                    <p><strong>Earthquake Spot:</strong> {result.analysis.funIdeas.earthquakeSpot}</p>
                </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}