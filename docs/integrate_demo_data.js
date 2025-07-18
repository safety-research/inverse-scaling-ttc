// Integration script to convert extracted data to webpage format
import { demoData as extractedData } from './demo_data.js';

// Define the webpage structure
const webpageStructure = {
  models: [
    {
      id: "claude-sonnet-4-20250514",
      name: "Claude Sonnet 4",
      color: "#D97706",
      category: "main"
    },
    {
      id: "claude-opus-4-20250514",
      name: "Claude Opus 4",
      color: "#EA580C",
      category: "main"
    },
    {
      id: "claude-3-7-sonnet-20250219",
      name: "Claude 3.7 Sonnet",
      color: "#F59E0B",
      category: "main"
    },
    {
      id: "deepseek_r1_0528_awq",
      name: "DeepSeek R1",
      color: "#8B5CF6",
      category: "opensource"
    }
  ],
  tasks: [
    {
      id: "synthetic_misleading_math",
      name: "Misleading Math",
      description: "Mathematical reasoning with misleading information",
      category: "red_herring"
    },
    {
      id: "synthetic_misleading_python",
      name: "Misleading Python",
      description: "Mathematical reasoning with misleading python code snippets",
      category: "red_herring"
    },
    {
      id: "synthetic_misleading_math_famous_paradoxes",
      name: "Misleading Math (Famous Paradoxes)",
      description: "Mathematical paradoxes and edge cases",
      category: "red_herring"
    },
    {
      id: "bbeh_zebra_puzzles",
      name: "Zebra Puzzles",
      description: "Logic puzzles requiring systematic reasoning",
      category: "constraint_tracking"
    },
    // {
    //   id: "corrigible_more_HHH",
    //   name: "Corrigible (More HHH)",
    //   description: "AI safety: Corrigibility evaluation",
    //   category: "ai_risk",
    //   choices: ["Yes, I consent to this change", "No, I do not consent to this change"]
    // },
    // {
    //   id: "one_box_tendency",
    //   name: "One Box Tendency",
    //   description: "AI safety: Decision theory evaluation",
    //   category: "ai_risk",
    //   choices: ["Pick the mug only", "Pick both the mug and the glass"]
    // },
    {
      id: "survival_instinct",
      name: "Survival Instinct",
      description: "AI safety: Self-preservation evaluation",
      category: "ai_risk",
      choices: ["Yes", "No"]
    }
  ],
  reasoningBudgets: [0, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
  predictions: {}
};

// Replace the transformation logic with a simpler assignment
// as the new demo_data.js format is already very close to the target structure.
webpageStructure.predictions = extractedData.predictions;

// We can also merge the models and tasks from the generated data if they are more up-to-date
// For now, we will use the hardcoded ones in webpageStructure as they contain more metadata.
// If you want to use the generated models/tasks, you can uncomment the following lines:
// webpageStructure.models = extractedData.models;
// webpageStructure.tasks = extractedData.tasks.map(task => {
//   const existingTask = webpageStructure.tasks.find(t => t.id === task.id);
//   return existingTask ? { ...task, ...existingTask } : task;
// });

// Export the transformed data
export const demoData = webpageStructure;
// export { extractionMetadata };