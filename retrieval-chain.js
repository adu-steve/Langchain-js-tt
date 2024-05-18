import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import * as dotenv from "dotenv";
import { Document } from "@langchain/core/documents";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";

dotenv.config({});

const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo",
  temperature: 0.7,
});

const prompt = ChatPromptTemplate.fromTemplate(`
Answer the following question .
Context :{context}
Question:{input}`);

const documentA = new Document({
  pageContent:
    "Study skills or study strategies are approaches applied to learning. Study skills are an array of skills which tackle the process of organizing and taking in new information, retaining information, or dealing with assessments. They are discrete techniques that can be learned, usually in a short time, and applied to all or most fields of study. More broadly, any skill which boosts a person's ability to study, retain and recall information which assists in and passing exams can be termed a study skill, and this could include time management and motivational techniques.",
});
const documentB = new Document({
  pageContent:
    "Due to the generic nature of study skills, they must, therefore, be distinguished from strategies that are specific to a particular field of study (e.g. music or technology), and from abilities inherent in the student, such as aspects of intelligence or learning styles. It is crucial in this, however, for students to gain initial insight into their habitual approaches to study, so they may better understand the dynamics and personal resistances to learning new techniques.",
});

//const chain = prompt.pipe(model);
const chain = await createStuffDocumentsChain({
  llm: model,
  prompt,
});
const response = await chain.invoke({
  input: "What is study skills and provide other information about it",
  context: [documentA, documentB],
});
console.log(response);
