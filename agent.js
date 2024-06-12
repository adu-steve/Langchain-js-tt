import { ChatOpenAI } from "@langchain/openai";
import * as dotenv from "dotenv";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { AgentExecutor, createOpenAIFunctionsAgent } from "langchain/agents";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import readline from "readline";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
dotenv.config();

const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo-1106",
  temperature: 0.7,
});

const prompt = ChatPromptTemplate.fromMessages([
  ("system", "You are a helpful assistant called Max"),
  new MessagesPlaceholder("chat_history"),
  ("human", "{input}"),
  new MessagesPlaceholder("agent_scratchpad"),
]);

//CERATE AND ASSIGN TOOLS
const searchTool = new TavilySearchResults();
const tools = [searchTool];
//create agent
const agent = await createOpenAIFunctionsAgent({
  llm: model,
  prompt,
  tools,
});

//create agent executor
const agentExecutor = new AgentExecutor({
  agent,
  tools,
});

//taking input from the user
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const chathistory = [];

const askQuestion = () => {
  rl.question("User : ", async (input) => {
    if (input.toLowerCase() === "exit") {
      rl.close();
      return;
    }
    const response = await agentExecutor.invoke({
      input: input,
      chat_history: chathistory,
    });

    console.log("Agent : " + response.output);
    chathistory.push(new HumanMessage(input));
    chathistory.push(new AIMessage(response.output));
    askQuestion();
  });
};
askQuestion();
