<script setup lang="ts">
import { onMounted, ref } from 'vue'
import LlmView from '@/views/LlmView.vue'
import VisionView from '@/views/VisionView.vue'
import DiffusionView from '@/views/DiffusionView.vue'
import { backendInfo, type BackendInfo } from '@/lib/tauri'

type Tab = 'llm' | 'vision' | 'diffusion'
const tab = ref<Tab>('llm')
const info = ref<BackendInfo | null>(null)

onMounted(async () => {
  try {
    info.value = await backendInfo()
  } catch (e) {
    console.error('backend_info failed', e)
  }
})

const tabs: Array<{ id: Tab; label: string }> = [
  { id: 'llm', label: 'LLM' },
  { id: 'vision', label: 'Vision' },
  { id: 'diffusion', label: 'Diffusion' },
]
</script>

<template>
  <div class="flex h-full flex-col">
    <header class="flex items-center justify-between border-b border-zinc-800 bg-zinc-900 px-4 py-2">
      <div class="flex items-center gap-3">
        <h1 class="text-sm font-semibold tracking-tight">scry-studio</h1>
        <span v-if="info" class="text-xs text-zinc-400">
          backend: <span class="text-zinc-200">{{ info.name }}</span>
          <span v-if="info.cuda" class="ml-2 rounded bg-emerald-900/40 px-1 text-emerald-300">cuda</span>
          <span v-if="info.bf16" class="ml-1 rounded bg-emerald-900/40 px-1 text-emerald-300">bf16</span>
          <span v-if="info.onnx" class="ml-1 rounded bg-blue-900/40 px-1 text-blue-300">onnx</span>
        </span>
      </div>
      <nav class="flex gap-1">
        <button
          v-for="t in tabs"
          :key="t.id"
          class="btn"
          :class="{ 'btn-primary': tab === t.id }"
          @click="tab = t.id"
        >
          {{ t.label }}
        </button>
      </nav>
    </header>

    <main class="flex-1 overflow-auto p-4">
      <LlmView v-if="tab === 'llm'" />
      <VisionView v-else-if="tab === 'vision'" />
      <DiffusionView v-else-if="tab === 'diffusion'" />
    </main>
  </div>
</template>
