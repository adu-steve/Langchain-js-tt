import { ChatOpenAI } from "@langchain/openai";
import * as dotenv from "dotenv";
import { ChatPromptTemplate } from "@langchain/core/prompts";

dotenv.config();

const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo",
  temperature: 0.7,
});

const prompt = ChatPromptTemplate.fromMessages([
  ["system", "Generate a poem from the word provided by the user"],
  ["human", "{input}"],
]);

const chain = prompt.pipe(model);
const response = await chain.invoke({ input: "dragon" });
console.log(response);
