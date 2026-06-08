import prompt001 from "./canonicalPrompts/001.json";
import prompt005 from "./canonicalPrompts/005.json";
import prompt007 from "./canonicalPrompts/007.json";
import prompt009 from "./canonicalPrompts/009.json";
import prompt021 from "./canonicalPrompts/021.json";
import prompt022 from "./canonicalPrompts/022.json";

const prettyJson = (value: unknown) => JSON.stringify(value, null, 2);

export const CANONICAL_LONG_PROMPTS = {
  "001": prettyJson(prompt001),
  "005": prettyJson(prompt005),
  "007": prettyJson(prompt007),
  "009": prettyJson(prompt009),
  "021": prettyJson(prompt021),
  "022": prettyJson(prompt022)
} as const;
