import { ChatOpenAI } from "@langchain/openai";
import * as dotenv from "dotenv";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import {
  StringOutputParser,
  CommaSeparatedListOutputParser,
} from "@langchain/core/output_parsers";
import { StructuredOutputParser } from "langchain/output_parsers";
import { z } from "zod";

dotenv.config();

const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo",
  temperature: 0.7,
});
//a function of stringOutput Parser
async function callStringOutputParser() {
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "Generate a love anthem from the word provided by the user"],
    ["human", "{input}"],
  ]);

  const parser = new StringOutputParser();

  const chain = prompt.pipe(model).pipe(parser);

  return await chain.invoke({ input: "dragon" });
}
// a function of the list or comma seperated output parser.
async function callCommaSeparatedOutputParser() {
  const prompt = ChatPromptTemplate.fromTemplate(
    "Generate five synonyms from the word {input} separated by commas"
  );

  const listoutputparser = new CommaSeparatedListOutputParser();

  const chain = prompt.pipe(model).pipe(listoutputparser);
  return await chain.invoke({ input: "dragon" });
}

//a function for production based for outputting a parser called StructuredOutputParser
async function callStructuredParser() {
  const prompt = ChatPromptTemplate.fromTemplate(
    `Extract information from the following phrase 
    Formatting-Instructions :{format_instructions}
    Phrase:{phrase}`
  );

  const outputParser = StructuredOutputParser.fromNamesAndDescriptions({
    name: "the name of the person",
    age: "the age of the person",
  });
  const chain = prompt.pipe(model).pipe(outputParser);

  return await chain.invoke({
    phrase: "Stephen is 30 years old.",
    format_instructions: outputParser.getFormatInstructions(),
  });
}

//Another structured output parser using the zod Schema
const callZodOutputParser = async () => {
  const prompt = ChatPromptTemplate.fromTemplate(
    `Extract information from the following phrase 
    Formatting-Instructions :{format_instructions}
    Phrase:{phrase}`
  );

  const zodOutputParser = StructuredOutputParser.fromZodSchema(
    z.object({
      name: z.string().describe("name of the recipe"),
      ingredients: z.array(z.string().describe("ingredients of of the recipe")),
    })
  );
  const chain = prompt.pipe(model).pipe(zodOutputParser);

  return await chain.invoke({
    phrase:
      "I made cheese burger with the ingredients tomato, cheese, beef and flour",
    format_instructions: zodOutputParser.getFormatInstructions(),
  });
};
const response = await callZodOutputParser();
console.log(response);
