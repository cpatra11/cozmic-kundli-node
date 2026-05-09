import fs from 'fs';
import path from 'path';

const DEBUG_FILE = path.join(process.cwd(), 'debug-log.txt');

export function clearDebugLog(): void {
  fs.writeFileSync(DEBUG_FILE, '');
}

export function appendDebugLog(section: string, title: string, content: string): void {
  const timestamp = new Date().toISOString();
  const logEntry = `\n${'='.repeat(60)}
[${timestamp}] ${section}: ${title}
${'='.repeat(60)}
${content}
${'='.repeat(60)}\n`;

  fs.appendFileSync(DEBUG_FILE, logEntry);
}

export function logAgentStart(input: { message: string; mode: string; hasKundli: boolean; sessionId?: string }): void {
  clearDebugLog();
  appendDebugLog('AGENT', 'START', JSON.stringify(input, null, 2));
}

export function logNodeStart(nodeName: string, input: any): void {
  appendDebugLog(nodeName, 'INPUT', typeof input === 'string' ? input : JSON.stringify(input, null, 2));
}

export function logLLMCall(systemPrompt: string, userPrompt: string, response: string): void {
  appendDebugLog('LLM', 'SYSTEM_PROMPT', systemPrompt);
  appendDebugLog('LLM', 'USER_PROMPT', userPrompt);
  appendDebugLog('LLM', 'RAW_RESPONSE', response);
}

export function logNodeEnd(nodeName: string, output: any): void {
  appendDebugLog(nodeName, 'OUTPUT', typeof output === 'string' ? output : JSON.stringify(output, null, 2));
}