import { ChatOpenAI } from "@langchain/openai";
import * as dotenv from "dotenv";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { BufferMemory } from "langchain/memory";
import { ConversationChain } from "langchain/chains";

dotenv.config();

const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo",
  temperature: 0.7,
});

const prompt = ChatPromptTemplate.fromTemplate(`
You are an AI Assistant.
{history}
{input}`);

const memory = new BufferMemory({
  memoryKey: "history",
});

const chain = new ConversationChain({
  llm: model,
  prompt,
  memory,
});

console.log(await memory.loadMemoryVariables());
const input1 = { input: "The passphrase is HelloWorld" };

const response1 = await chain.invoke(input1);
console.log(response1);

console.log(await memory.loadMemoryVariables());
const input2 = { input: "The passphrase is what?" };

const response2 = await chain.invoke(input2);
console.log(response2);
