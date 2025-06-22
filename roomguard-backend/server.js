const express = require('express');
const multer = require('multer');
const cors = require('cors');
const axios = require('axios');
const dotenv = require('dotenv');

dotenv.config();

const app = express();
const port = 3001;

app.use(cors({
  origin: 'http://localhost:3000'
}));

const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const GEMINI_API_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${GEMINI_API_KEY}`;

if (!GEMINI_API_KEY) {
  console.error("Error: GEMINI_API_KEY not found in .env file. Please set it.");
  process.exit(1);
}

app.post('/api/analyze', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image file uploaded.' });
    }

    const base64Image = req.file.buffer.toString('base64');
    const mimeType = req.file.mimetype;

    const requestBody = {
      contents: [
        {
          parts: [
            {
              text: "Analyze this room for safety hazards, identify potential risks like fire hazards, tripping hazards, poor lighting, or improper equipment storage. For each hazard, provide a category (e.g., 'Fire', 'Tripping', 'Electrical', 'Structural', 'Chemical'), a severity level (e.g., 'High', 'Medium', 'Low'), a brief description, a recommended remediation, and a relevant (mock) OSHA standard if applicable. Also, provide approximate x and y coordinates (as percentages, 0-100) for each hazard's location on the image. Finally, give an overall safety score out of 100 for the room, and individual safety scores for categories like 'Electrical Safety', 'General Orderliness', 'Structural Integrity', and 'Lighting'. The output should be a JSON object adhering to the following structure:\n\n{\n  \"scores\": {\n    \"overall\": 85,\n    \"categories\": {\n      \"ElectricalSafety\": 70,\n      \"GeneralOrderliness\": 90,\n      \"StructuralIntegrity\": 80,\n      \"Lighting\": 95\n    }\n  },\n  \"hazards\": [\n    {\n      \"id\": \"h1\",\n      \"category\": \"Electrical\",\n      \"severity\": \"High\",\n      \"description\": \"Frayed electrical wire near a water source.\",\n      \"remediation\": \"Replace the frayed wire immediately and ensure proper insulation.\",\n      \"oshaStandard\": \"OSHA 1910.303(b)(1)\",\n      \"coordinates\": {\n        \"x\": 30,\n        \"y\": 45\n      }\n    },\n    {\n      \"id\": \"h2\",\n      \"category\": \"Tripping\",\n      \"severity\": \"Medium\",\n      \"description\": \"Box left in walkway, potential tripping hazard.\",\n      \"remediation\": \"Clear the walkway and store the box in designated storage area.\",\n      \"oshaStandard\": \"OSHA 1910.22(a)(1)\",\n      \"coordinates\": {\n        \"x\": 70,\n        \"y\": 80\n      }\n    }\n  ]\n}",
            },
            {
              inlineData: {
                mimeType: mimeType,
                data: base64Image,
              },
            },
          ],
        },
      ],
    };

    console.log("Forwarding image to Gemini API...");

    const geminiResponse = await axios.post(GEMINI_API_URL, requestBody, {
      headers: {
        'Content-Type': 'application/json',
      },
    });

    let textResponse = geminiResponse.data.candidates[0].content.parts[0].text;
    console.log("Gemini API raw response:", textResponse);

    // --- NEW FIX START ---
    // Remove Markdown code block delimiters if present
    if (textResponse.startsWith('```json')) {
      textResponse = textResponse.substring(7); // Remove '```json'
    }
    if (textResponse.endsWith('```')) {
      textResponse = textResponse.substring(0, textResponse.length - 3); // Remove '```'
    }
    textResponse = textResponse.trim(); // Remove any leading/trailing whitespace/newlines
    // --- NEW FIX END ---

    // Attempt to parse the AI's response as JSON
    const analysisResult = JSON.parse(textResponse);

    res.json(analysisResult);

  } catch (error) {
    console.error('Error analyzing image with Gemini API:', error.response ? error.response.data : error.message);
    if (error.response && error.response.data) {
        console.error('Gemini API Error Details:', error.response.data);
    }
    // Provide more specific error message if it's a JSON parsing issue
    if (error instanceof SyntaxError) {
        res.status(500).json({ error: 'Failed to parse AI response as JSON. The AI might not have returned the expected format.', details: error.message, rawResponse: textResponse });
    } else {
        res.status(500).json({ error: 'Failed to analyze image. Please try again.', details: error.message });
    }
  }
});

app.listen(port, () => {
  console.log(`Backend server listening at http://localhost:${port}`);
});