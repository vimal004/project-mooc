const { GoogleGenerativeAI } = require("@google/generative-ai");

const genAI = new GoogleGenerativeAI("AIzaSyAhLSNxek1FO343UFH1pDNKoAXFDUN8L1g");
const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

const prompt = "Tomato___Tomato_mosaic_virus signs syptoms and control measures prevention techniques point wise short sweet crisp";

// Wrap the async call in an async function
async function generateContent() {
  try {
    const result = await model.generateContent(prompt);
    console.log(result.response.text());
  } catch (error) {
    console.error("Error generating content:", error);
  }
}

generateContent();
