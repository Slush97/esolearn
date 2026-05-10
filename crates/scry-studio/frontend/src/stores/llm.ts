import { defineStore } from 'pinia'
import { ref } from 'vue'
import {
  llmDecode,
  llmGenerate,
  llmLoad,
  llmLogits,
  llmTokenize,
  type LlmGenerateArgs,
  type LlmLoadArgs,
  type LlmLoadResult,
  type LogitsResult,
  type TokenizeResult,
} from '@/lib/tauri'

export const useLlmStore = defineStore('llm', () => {
  const loaded = ref<LlmLoadResult | null>(null)
  const loading = ref(false)
  const generating = ref(false)
  const lastOutput = ref<string>('')
  const lastElapsedMs = ref(0)
  const tokenization = ref<TokenizeResult | null>(null)
  const logits = ref<LogitsResult | null>(null)
  const error = ref<string | null>(null)

  async function load(args: LlmLoadArgs) {
    loading.value = true
    error.value = null
    try {
      loaded.value = await llmLoad(args)
    } catch (e) {
      error.value = String(e)
    } finally {
      loading.value = false
    }
  }

  async function generate(args: LlmGenerateArgs) {
    if (!loaded.value) {
      error.value = 'load a model first'
      return
    }
    generating.value = true
    error.value = null
    try {
      const r = await llmGenerate(args)
      lastOutput.value = r.text
      lastElapsedMs.value = r.elapsedMs
    } catch (e) {
      error.value = String(e)
    } finally {
      generating.value = false
    }
  }

  async function tokenize(text: string) {
    error.value = null
    try {
      tokenization.value = await llmTokenize(text)
    } catch (e) {
      error.value = String(e)
    }
  }

  async function decode(ids: number[]) {
    error.value = null
    try {
      const r = await llmDecode(ids)
      lastOutput.value = r.text
    } catch (e) {
      error.value = String(e)
    }
  }

  async function inspectLogits(prompt: string, topK = 20) {
    error.value = null
    try {
      logits.value = await llmLogits(prompt, topK)
    } catch (e) {
      error.value = String(e)
    }
  }

  return {
    loaded,
    loading,
    generating,
    lastOutput,
    lastElapsedMs,
    tokenization,
    logits,
    error,
    load,
    generate,
    tokenize,
    decode,
    inspectLogits,
  }
})
