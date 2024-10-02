import 'dotenv/config';

import { GoogleGenerativeAI } from '@google/generative-ai';
import { HfInference } from '@huggingface/inference';
import axios from 'axios';
import { Embedding, STSDataset, STSRecord } from './types';

const geminiClient = new GoogleGenerativeAI(
  process.env.GOOGLE_API_KEY || ''
).getGenerativeModel({ model: 'text-embedding-004' });
const hfClient = new HfInference(process.env.HUGGINGFACE_API_KEY || '');

// Function to get embeddings from Google Gemini
async function getGeminiEmbeddings(texts: string[]): Promise<Embedding[]> {
  const responses = await Promise.all(
    texts.map((text) => geminiClient.embedContent(text))
  );
  return responses.map((res) => res.embedding.values);
}

const geminiSums = { sumX: 0, sumY: 0, sumXY: 0, sumX2: 0, sumY2: 0, n: 0 };
const hfSums = { sumX: 0, sumY: 0, sumXY: 0, sumX2: 0, sumY2: 0, n: 0 };

// Function to get embeddings from Hugging Face
async function getHuggingFaceEmbeddings(
  texts: string[],
  model = 'sentence-transformers/all-MiniLM-L6-v2'
): Promise<Embedding[]> {
  const responses = await hfClient.featureExtraction({
    model: model,
    inputs: texts,
  });

  return responses as Embedding[];
}

// Function to compute cosine similarity between two vectors
function cosineSimilarity(vecA: Embedding, vecB: Embedding) {
  const dotProduct = vecA.reduce((sum, a, index) => sum + a * vecB[index], 0);
  const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));

  return dotProduct / (magnitudeA * magnitudeB);
}

// Update aggregate sums for each chunk of the dataset
function updateSums(obj: any, similarities: number[], trueScores: number[]) {
  for (let i = 0; i < similarities.length; i++) {
    const X = similarities[i];
    const Y = trueScores[i] / 5;

    obj.sumX += X;
    obj.sumY += Y;
    obj.sumXY += X * Y;
    obj.sumX2 += X * X;
    obj.sumY2 += Y * Y;
    obj.n += 1;
  }
}

// Compute overall Pearson correlation using aggregate sums
function calculatePearsonCorrelation(obj: any) {
  const numerator = obj.n * obj.sumXY - obj.sumX * obj.sumY;
  const denominator = Math.sqrt(
    (obj.n * obj.sumX2 - obj.sumX * obj.sumX) *
      (obj.n * obj.sumY2 - obj.sumY * obj.sumY)
  );

  return numerator / denominator;
}

// Function to download and parse the STS Benchmark dataset
async function loadSTSData(offset = 0): Promise<STSRecord[]> {
  try {
    const response = await axios.get<STSDataset>(
      'https://datasets-server.huggingface.co/rows?dataset=anti-ai%2FViSTS&config=STS-Sickr&split=test',
      {
        params: {
          offset,
          length: 100,
        },
      }
    );

    return response.data.rows.map(({ row }) => row);
  } catch (error) {
    console.error(error);
    return [];
  }
}

// Main function for benchmarking
async function benchmark() {
  for (let i = 0; i < 3; i++) {
    for (let offset = 0; offset < 700; offset += 100) {
      console.log(`Processing from offset: ${i * 700 + offset}...`);

      const stsData = await loadSTSData(offset);
      const sentences1 = stsData.map((item) => item.sentence1);
      const sentences2 = stsData.map((item) => item.sentence2);
      const trueScores = stsData.map((item) => item.score);

      // Get embeddings using Google Gemini and Hugging Face
      try {
        const geminiEmbeds1 = await getGeminiEmbeddings(sentences1);
        const geminiEmbeds2 = await getGeminiEmbeddings(sentences2);
        const hfEmbeds1 = await getHuggingFaceEmbeddings(sentences1);
        const hfEmbeds2 = await getHuggingFaceEmbeddings(sentences2);

        const geminiSimilarities = geminiEmbeds1.map((emb, index) =>
          cosineSimilarity(emb, geminiEmbeds2[index])
        );
        const hfSimilarities = hfEmbeds1.map((emb, index) =>
          cosineSimilarity(emb, hfEmbeds2[index])
        );

        updateSums(geminiSums, geminiSimilarities, trueScores);
        updateSums(hfSums, hfSimilarities, trueScores);
      } catch (error) {
        console.log('Embedding failed:', error);
        break;
      }
    }

    await new Promise(f => setTimeout(f, 60000));
  }

  console.log('Calculating Pearson Correlation...');

  console.log(
    `Gemini Model Pearson Correlation: ${calculatePearsonCorrelation(
      geminiSums
    ).toFixed(4)}`
  );
  console.log(
    `Hugging Face Pearson Correlation: ${calculatePearsonCorrelation(
      hfSums
    ).toFixed(4)}`
  );

  console.log('DONE!');
}

// Run the benchmark
benchmark().catch(console.error);
