import { invoke } from '@tauri-apps/api/core'

export interface BackendInfo {
  name: string
  onnx: boolean
  cuda: boolean
  bf16: boolean
}

export async function backendInfo(): Promise<BackendInfo> {
  return invoke<BackendInfo>('backend_info')
}

// ---------- LLM ----------

export interface LlmLoadArgs {
  modelPath: string
  tokenizerPath: string
  preset: 'gpt2'
}

export interface LlmLoadResult {
  vocabSize: number
  maxSeqLen: number
  nLayers: number
  dModel: number
}

export async function llmLoad(args: LlmLoadArgs): Promise<LlmLoadResult> {
  return invoke<LlmLoadResult>('llm_load', {
    args: {
      model_path: args.modelPath,
      tokenizer_path: args.tokenizerPath,
      preset: args.preset,
    },
  }).then((r) => snakeToCamel(r) as LlmLoadResult)
}

export interface LlmGenerateArgs {
  prompt: string
  maxTokens?: number
  temperature?: number
  topK?: number
  topP?: number
  seed?: number
}

export interface LlmGenerateResult {
  text: string
  promptTokens: number
  generatedTokens: number
  elapsedMs: number
}

export async function llmGenerate(args: LlmGenerateArgs): Promise<LlmGenerateResult> {
  return invoke<LlmGenerateResult>('llm_generate', {
    args: {
      prompt: args.prompt,
      max_tokens: args.maxTokens,
      temperature: args.temperature,
      top_k: args.topK,
      top_p: args.topP,
      seed: args.seed,
    },
  }).then((r) => snakeToCamel(r) as LlmGenerateResult)
}

export interface TokenizeResult {
  ids: number[]
  pieces: string[]
}

export async function llmTokenize(text: string): Promise<TokenizeResult> {
  return invoke<TokenizeResult>('llm_tokenize', { args: { text } })
}

export async function llmDecode(ids: number[]): Promise<{ text: string }> {
  return invoke<{ text: string }>('llm_decode', { args: { ids } })
}

export interface TopToken {
  id: number
  piece: string
  logit: number
  prob: number
}

export interface LogitsResult {
  promptTokens: number[]
  top: TopToken[]
}

export async function llmLogits(prompt: string, topK = 20): Promise<LogitsResult> {
  return invoke<LogitsResult>('llm_logits', {
    args: { prompt, top_k: topK },
  }).then((r) => snakeToCamel(r) as LogitsResult)
}

// ---------- Vision ----------

export interface ResnetLoadArgs {
  modelPath: string
  preset: 'resnet18' | 'resnet34' | 'resnet50' | 'resnet101' | 'resnet152'
  labelsPath?: string
}

export interface ResnetLoadResult {
  numClasses: number
  hasLabels: boolean
}

export async function resnetLoad(args: ResnetLoadArgs): Promise<ResnetLoadResult> {
  return invoke<ResnetLoadResult>('resnet_load', {
    args: {
      model_path: args.modelPath,
      preset: args.preset,
      labels_path: args.labelsPath,
    },
  }).then((r) => snakeToCamel(r) as ResnetLoadResult)
}

export interface ResnetClassifyEntry {
  classId: number
  label: string | null
  score: number
}

export interface ResnetClassifyResult {
  top: ResnetClassifyEntry[]
  elapsedMs: number
}

export async function resnetClassify(bytes: Uint8Array, topK = 5): Promise<ResnetClassifyResult> {
  return invoke<ResnetClassifyResult>('resnet_classify', {
    args: { image: { bytes: Array.from(bytes) }, top_k: topK },
  }).then((r) => snakeToCamel(r) as ResnetClassifyResult)
}

export interface DetectionDto {
  bbox: [number, number, number, number]
  classId: number
  confidence: number
  keypoints: Array<[number, number]> | null
}

export interface ScrfdDetectResult {
  detections: DetectionDto[]
  elapsedMs: number
}

export async function scrfdLoad(modelPath: string, inputSize = 640): Promise<{ inputSize: number }> {
  return invoke<{ inputSize: number }>('scrfd_load', {
    args: { model_path: modelPath, input_size: inputSize },
  }).then((r) => snakeToCamel(r) as { inputSize: number })
}

export async function scrfdDetect(bytes: Uint8Array, conf = 0.5): Promise<ScrfdDetectResult> {
  return invoke<ScrfdDetectResult>('scrfd_detect', {
    args: { image: { bytes: Array.from(bytes) }, conf_threshold: conf },
  }).then((r) => snakeToCamel(r) as ScrfdDetectResult)
}

// ---------- Diffusion ----------

export async function diffusionLoad(snapshotPath: string): Promise<{ backend: string }> {
  return invoke<{ backend: string }>('diffusion_load', {
    args: { snapshot_path: snapshotPath },
  })
}

export interface DiffusionGenerateArgs {
  prompt: string
  negativePrompt?: string
  steps?: number
  cfg?: number
  seed?: number
  size?: number
}

export interface DiffusionGenerateResult {
  png: number[]
  width: number
  height: number
  elapsedMs: number
}

export async function diffusionGenerate(
  args: DiffusionGenerateArgs,
): Promise<DiffusionGenerateResult> {
  return invoke<DiffusionGenerateResult>('diffusion_generate', {
    args: {
      prompt: args.prompt,
      negative_prompt: args.negativePrompt ?? '',
      steps: args.steps,
      cfg: args.cfg,
      seed: args.seed,
      size: args.size,
    },
  }).then((r) => snakeToCamel(r) as DiffusionGenerateResult)
}

// Tauri serializes Rust structs with snake_case fields by default; rather than
// adding a serde rename to every DTO, we shallow-convert keys here.
function snakeToCamel<T = unknown>(value: unknown): T {
  if (Array.isArray(value)) return value.map((v) => snakeToCamel(v)) as unknown as T
  if (value && typeof value === 'object') {
    const out: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      const ck = k.replace(/_([a-z])/g, (_, c) => c.toUpperCase())
      out[ck] = snakeToCamel(v)
    }
    return out as T
  }
  return value as T
}
