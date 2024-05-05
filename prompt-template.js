import { ChatOpenAI } from "@langchain/openai";
import * as dotenv from "dotenv";
import { ChatPromptTemplate } from "langchain/core/prompts";

dotenv.config();

const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo",
  temperature: 0.7,
});

const prompt = ChatPromptTemplate.fromTemplate(
  "You're a comedian. Tell a joke based on the following word {word}"
);
console.log(await prompt.format({ word: "mouse" }));
