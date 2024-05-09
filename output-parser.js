import { ChatOpenAI } from "@langchain/openai";
import * as dotenv from "dotenv";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import {
  StringOutputParser,
  CommaSeparatedListOutputParser,
} from "@langchain/core/output_parsers";

dotenv.config();

const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo",
  temperature: 0.7,
});
//a function of stringOutput Parser
// async function callStringOutputParser() {
//   const prompt = ChatPromptTemplate.fromMessages([
//     ["system", "Generate a poem from the word provided by the user"],
//     ["human", "{input}"],
//   ]);

//   const parser = new StringOutputParser();

//   const chain = prompt.pipe(model).pipe(parser);

//   return await chain.invoke({ input: "dragon" });
// }

async function callCommaSeparatedOutputParser() {
  const prompt = ChatPromptTemplate.fromTemplate(
    "Generate five synonyms from the word {input}"
  );

  const stringOutputParser = new StringOutputParser();
  const listoutputparser = new CommaSeparatedListOutputParser();

  const chain = prompt
    .pipe(model)
    .pipe(stringOutputParser)
    .pipe(listoutputparser);

  return await chain.invoke({ input: "dragon" });
}

const response = await callCommaSeparatedOutputParser();
console.log(response);
