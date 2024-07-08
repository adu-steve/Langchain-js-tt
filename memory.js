import { ChatOpenAI } from "@langchain/openai";
import * as dotenv from "dotenv";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { BufferMemory } from "langchain/memory";
import { ConversationChain } from "langchain/chains";
import { UpstashRedisChatMessageHistory } from "@langchain/community/stores/message/upstash_redis";

dotenv.config();

const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo",
  temperature: 0.7,
});

const prompt = ChatPromptTemplate.fromTemplate(`
You are an AI Assistant.
{history}
{input}`);

const upstashChatHistory = new UpstashRedisChatMessageHistory({
  sessionId: "chat1",
  config: {
    url: "https://delicate-shepherd-53256.upstash.io",
    token: "AdAIAAIncDExYmVkNGY3Nzc4NWI0YmIxOThmZWNhMGU1NmI0ZWE0ZXAxNTMyNTY",
  },
});
const memory = new BufferMemory({
  memoryKey: "history",
  chatHistory: upstashChatHistory,
});

const chain = new ConversationChain({
  llm: model,
  prompt,
  memory,
});

const input2 = { input: "What was the question I just asked?" };

const response2 = await chain.invoke(input2);
console.log(response2);
