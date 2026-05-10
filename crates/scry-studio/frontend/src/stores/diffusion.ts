import { defineStore } from 'pinia'
import { ref } from 'vue'
import {
  diffusionGenerate,
  diffusionLoad,
  type DiffusionGenerateArgs,
} from '@/lib/tauri'

export const useDiffusionStore = defineStore('diffusion', () => {
  const loaded = ref<{ backend: string } | null>(null)
  const loading = ref(false)
  const generating = ref(false)
  const imageUrl = ref<string | null>(null)
  const lastElapsedMs = ref(0)
  const error = ref<string | null>(null)

  async function load(snapshotPath: string) {
    loading.value = true
    error.value = null
    try {
      loaded.value = await diffusionLoad(snapshotPath)
    } catch (e) {
      error.value = String(e)
    } finally {
      loading.value = false
    }
  }

  async function generate(args: DiffusionGenerateArgs) {
    if (!loaded.value) {
      error.value = 'load the SD snapshot first'
      return
    }
    generating.value = true
    error.value = null
    try {
      const r = await diffusionGenerate(args)
      const blob = new Blob([new Uint8Array(r.png)], { type: 'image/png' })
      if (imageUrl.value) URL.revokeObjectURL(imageUrl.value)
      imageUrl.value = URL.createObjectURL(blob)
      lastElapsedMs.value = r.elapsedMs
    } catch (e) {
      error.value = String(e)
    } finally {
      generating.value = false
    }
  }

  return { loaded, loading, generating, imageUrl, lastElapsedMs, error, load, generate }
})
