#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import OpenAI from "openai";

const API_KEY = process.env.GOOGLE_API_KEY;
const NATIVE_API_BASE = "https://generativelanguage.googleapis.com/v1beta";

// OpenAI-compatible client for chat completions
const gemini = new OpenAI({
  apiKey: API_KEY,
  baseURL: "https://generativelanguage.googleapis.com/v1beta/openai/",
});

// Create MCP server
const server = new Server(
  {
    name: "gemini-mcp",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// List available tools
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "ask_gemini",
        description:
          "Delegate tasks to Google Gemini 3 Flash for general analysis, synthesis, summarization, and reasoning tasks. Fast and cost-effective for most delegation needs.",
        inputSchema: {
          type: "object",
          properties: {
            prompt: {
              type: "string",
              description:
                "The task/question to send to Gemini. Be specific and detailed.",
            },
            system_prompt: {
              type: "string",
              description:
                "Optional system prompt to guide Gemini's behavior (e.g., 'You are an expert Python developer')",
            },
            temperature: {
              type: "number",
              description:
                "Temperature for response randomness (0.0-1.0). Default: 0.7",
              default: 0.7,
            },
            max_tokens: {
              type: "number",
              description: "Maximum tokens in response. Default: 8192",
              default: 8192,
            },
          },
          required: ["prompt"],
        },
      },
      {
        name: "ask_gemini_pro",
        description:
          "Delegate to Google Gemini 3 Pro for complex reasoning, code generation, architecture design, and demanding cognitive tasks. Most capable model.",
        inputSchema: {
          type: "object",
          properties: {
            prompt: {
              type: "string",
              description:
                "The complex task/question to send to Gemini Pro",
            },
            system_prompt: {
              type: "string",
              description:
                "Optional system prompt to guide Gemini Pro's behavior",
            },
            temperature: {
              type: "number",
              description:
                "Temperature for response randomness (0.0-1.0). Default: 0.7",
              default: 0.7,
            },
            max_tokens: {
              type: "number",
              description: "Maximum tokens in response. Default: 16384",
              default: 16384,
            },
          },
          required: ["prompt"],
        },
      },
      {
        name: "web_search",
        description:
          "LLM-optimized web search using Gemini with Google Search grounding. Returns structured search results with titles, URLs, and summaries. Use for competitive intelligence, market research, and real-time information.",
        inputSchema: {
          type: "object",
          properties: {
            search_query: {
              type: "string",
              description: "The search query. Be specific and detailed.",
            },
            count: {
              type: "number",
              description:
                "Number of results to return (1-50). Default: 10. Use 50 for deep research, 5 for quick checks.",
              default: 10,
              minimum: 1,
              maximum: 50,
            },
            search_recency_filter: {
              type: "string",
              description:
                "Time range filter. Options: oneDay (breaking news), oneWeek (recent), oneMonth (trends), oneYear (annual), noLimit (all time). Default: noLimit",
              enum: [
                "oneDay",
                "oneWeek",
                "oneMonth",
                "oneYear",
                "noLimit",
              ],
              default: "noLimit",
            },
            search_domain_filter: {
              type: "string",
              description:
                "Whitelist specific domains (e.g., 'techcrunch.com,venturebeat.com'). Optional.",
            },
          },
          required: ["search_query"],
        },
      },
      {
        name: "web_reader",
        description:
          "Fetch and parse full content from a specific URL using Gemini's URL context capability. Returns markdown-formatted content ready for analysis. Use to read articles, documentation, competitor pages.",
        inputSchema: {
          type: "object",
          properties: {
            url: {
              type: "string",
              description:
                "The URL to fetch and parse. Must be a valid http/https URL.",
            },
            return_format: {
              type: "string",
              description:
                "Content format. Options: markdown (default, best for LLM), text (plain text). Default: markdown",
              enum: ["markdown", "text"],
              default: "markdown",
            },
            with_images_summary: {
              type: "boolean",
              description:
                "Include summary of images found on page. Default: false",
              default: false,
            },
            with_links_summary: {
              type: "boolean",
              description:
                "Include summary of links found on page. Default: false",
              default: false,
            },
            timeout: {
              type: "number",
              description: "Request timeout in seconds. Default: 20",
              default: 20,
            },
          },
          required: ["url"],
        },
      },
      {
        name: "parse_document",
        description:
          "Extract text from documents, images, and PDFs using Gemini's multimodal capabilities. Handles complex layouts, tables, multi-column text. Use for PDF proposals/contracts, scanned documents, invoices. Supports up to 20MB files.",
        inputSchema: {
          type: "object",
          properties: {
            file_url: {
              type: "string",
              description:
                "URL of the file to parse (image or PDF). Must be a publicly accessible URL.",
            },
            return_format: {
              type: "string",
              description:
                "Output format. Options: markdown (default, best for LLM), text (plain text). Default: markdown",
              enum: ["markdown", "text"],
              default: "markdown",
            },
            parse_mode: {
              type: "string",
              description:
                "Parsing mode. Options: auto (automatic detection, default), ocr (force OCR), layout (preserve layout). Default: auto",
              enum: ["auto", "ocr", "layout"],
              default: "auto",
            },
          },
          required: ["file_url"],
        },
      },
    ],
  };
});

// Helper: build search prompt with filters
function buildSearchPrompt(query, count, recencyFilter, domainFilter) {
  let prompt = `Search for: ${query}. Return the top ${count || 10} results as a structured list. Each result should include: a title, the URL, and a brief summary of the content.`;

  if (recencyFilter && recencyFilter !== "noLimit") {
    const recencyMap = {
      oneDay: "the last 24 hours",
      oneWeek: "the last week",
      oneMonth: "the last month",
      oneYear: "the last year",
    };
    prompt += ` Focus on results from ${recencyMap[recencyFilter] || recencyFilter}.`;
  }

  if (domainFilter) {
    prompt += ` Only include results from these domains: ${domainFilter}.`;
  }

  return prompt;
}

// Helper: call native Gemini API
async function callNativeGemini(model, contents, tools, temperature, maxOutputTokens) {
  const url = `${NATIVE_API_BASE}/models/${model}:generateContent?key=${API_KEY}`;

  const body = {
    contents,
    generationConfig: {
      temperature: temperature || 0.1,
      maxOutputTokens: maxOutputTokens || 8192,
    },
  };

  if (tools) {
    body.tools = tools;
  }

  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Gemini API error: ${response.status} - ${errorText}`);
  }

  return await response.json();
}

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    // --- ask_gemini ---
    if (name === "ask_gemini") {
      const messages = [];
      if (args.system_prompt) {
        messages.push({ role: "system", content: args.system_prompt });
      }
      messages.push({ role: "user", content: args.prompt });

      const completion = await gemini.chat.completions.create({
        model: "gemini-3-flash-preview",
        messages,
        temperature: args.temperature || 0.7,
        max_tokens: args.max_tokens || 8192,
      });

      const content = completion.choices[0].message.content;
      if (!content) {
        throw new Error("Gemini returned empty response");
      }

      return {
        content: [{ type: "text", text: `Gemini Response:\n\n${content}` }],
      };
    }

    // --- ask_gemini_pro ---
    if (name === "ask_gemini_pro") {
      const messages = [];
      if (args.system_prompt) {
        messages.push({ role: "system", content: args.system_prompt });
      } else {
        messages.push({
          role: "system",
          content:
            "You are an expert software engineer. Provide clean, efficient, well-documented code with best practices. Focus on correctness, readability, and maintainability.",
        });
      }
      messages.push({ role: "user", content: args.prompt });

      const completion = await gemini.chat.completions.create({
        model: "gemini-3.1-pro-preview",
        messages,
        temperature: args.temperature || 0.7,
        max_tokens: args.max_tokens || 16384,
      });

      const content = completion.choices[0].message.content;
      if (!content) {
        throw new Error("Gemini Pro returned empty response");
      }

      return {
        content: [
          { type: "text", text: `Gemini Pro Response:\n\n${content}` },
        ],
      };
    }

    // --- web_search (native Gemini API with google_search grounding) ---
    if (name === "web_search") {
      const searchPrompt = buildSearchPrompt(
        args.search_query,
        args.count,
        args.search_recency_filter,
        args.search_domain_filter
      );

      const data = await callNativeGemini(
        "gemini-3-flash-preview",
        [{ parts: [{ text: searchPrompt }] }],
        [{ google_search: {} }],
        0.1,
        8192
      );

      // Extract text response
      const textContent =
        data.candidates?.[0]?.content?.parts?.[0]?.text || "";

      // Extract grounding metadata if available
      const groundingMeta = data.candidates?.[0]?.groundingMetadata;
      const chunks = groundingMeta?.groundingChunks || [];

      let formattedResults = `Web Search Results for: "${args.search_query}"\n`;

      if (chunks.length > 0) {
        formattedResults += `Found ${chunks.length} grounded sources\n\n`;
        chunks.forEach((chunk, index) => {
          const web = chunk.web || {};
          formattedResults += `[${index + 1}] ${web.title || "Untitled"}\n`;
          formattedResults += `URL: ${web.uri || "N/A"}\n\n`;
        });
        formattedResults += `---\n\nSynthesized Results:\n\n${textContent}`;
      } else {
        formattedResults += `\n\n${textContent}`;
      }

      return {
        content: [{ type: "text", text: formattedResults }],
      };
    }

    // --- web_reader (native Gemini API with url_context) ---
    if (name === "web_reader") {
      const format = args.return_format || "markdown";
      let readerPrompt = `Read and extract the full content from this URL: ${args.url}. Return the content as ${format}. Preserve the document structure including headings, lists, and code blocks.`;

      if (args.with_images_summary) {
        readerPrompt += " Include a summary of images found on the page.";
      }
      if (args.with_links_summary) {
        readerPrompt += " Include a summary of links found on the page.";
      }

      const data = await callNativeGemini(
        "gemini-3-flash-preview",
        [{ parts: [{ text: readerPrompt }] }],
        [{ url_context: {} }],
        0.1,
        8192
      );

      const textContent =
        data.candidates?.[0]?.content?.parts?.[0]?.text || "";

      if (!textContent) {
        throw new Error("Gemini returned empty response for URL reading");
      }

      let formattedContent = `Web Page Content from: ${args.url}\n\n---\n\n${textContent}`;

      return {
        content: [{ type: "text", text: formattedContent }],
      };
    }

    // --- parse_document (OpenAI-compat with multimodal input) ---
    if (name === "parse_document") {
      const format = args.return_format || "markdown";
      const modeInstruction =
        args.parse_mode === "ocr"
          ? "Use OCR to extract all text."
          : args.parse_mode === "layout"
          ? "Preserve the original layout and formatting as closely as possible."
          : "Extract all text from this document.";

      const completion = await gemini.chat.completions.create({
        model: "gemini-3-flash-preview",
        messages: [
          {
            role: "user",
            content: [
              {
                type: "image_url",
                image_url: { url: args.file_url },
              },
              {
                type: "text",
                text: `${modeInstruction} Format the output as ${format}. Preserve layout structure, tables, and formatting. Be thorough and accurate.`,
              },
            ],
          },
        ],
        temperature: 0.1,
        max_tokens: 8192,
      });

      const content = completion.choices[0].message.content;
      if (!content) {
        throw new Error("Gemini returned empty response for document parsing");
      }

      let formattedContent = `Document Parsing Results from: ${args.file_url}\nFormat: ${format}\n---\n\n${content}`;

      return {
        content: [{ type: "text", text: formattedContent }],
      };
    }

    throw new Error(`Unknown tool: ${name}`);
  } catch (error) {
    console.error(`[Gemini MCP Error] ${name}:`, error.message);
    return {
      content: [
        {
          type: "text",
          text: `Error calling Gemini (${name}): ${error.message}`,
        },
      ],
      isError: true,
    };
  }
});

// Start server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Gemini MCP Server running on stdio");
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
