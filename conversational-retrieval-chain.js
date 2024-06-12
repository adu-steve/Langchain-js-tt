import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import * as dotenv from "dotenv";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { MessagesPlaceholder } from "@langchain/core/prompts";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
dotenv.config();

//creating a vectorStore function
const createVectoreStore = async () => {
  const cheerio = new CheerioWebBaseLoader(
    "https://python.langchain.com/v0.1/docs/expression_language/"
  );

  const docs = await cheerio.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 400,
    chunkOverlap: 20,
  });

  const splitDocs = await splitter.splitDocuments(docs);

  const embeddings = new OpenAIEmbeddings();

  const vectorStore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
  );
  return vectorStore;
};

//creating a chain function
const createChain = async () => {
  const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 0.7,
  });

  const prompt = ChatPromptTemplate.fromMessages([
    ("system",
    "Answer the user's questions based on the following context:{context}"),
    new MessagesPlaceholder("chathistory"),
    ("user", "{input}"),
  ]);

  const chain = await createStuffDocumentsChain({
    llm: model,
    prompt,
  });

  const retriever = vectorStore.asRetriever({
    k: 2,
  });

  const conversationChain = await createRetrievalChain({
    combineDocsChain: chain,
    retriever,
  });

  return conversationChain;
};

const vectorStore = await createVectoreStore();
const chain = await createChain(vectorStore);

const chat = [
  new HumanMessage("Hello"),
  new AIMessage("Hi, how can I help you?"),
  new HumanMessage("My name is Stephen"),
  new AIMessage("Hi Stephen, how can I help you?"),
  new HumanMessage("The weather in Ghana is very hot"),
  new AIMessage("LCEL stands for Langchain Expression Language"),
];

const response = await chain.invoke({
  input: "What did I say about the weather in Ghana?",
  chathistory: chat,
});
console.log(response);
