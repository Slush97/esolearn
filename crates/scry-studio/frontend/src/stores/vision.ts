import { defineStore } from 'pinia'
import { ref } from 'vue'
import {
  resnetClassify,
  resnetLoad,
  scrfdDetect,
  scrfdLoad,
  type ResnetClassifyResult,
  type ResnetLoadArgs,
  type ResnetLoadResult,
  type ScrfdDetectResult,
} from '@/lib/tauri'

export const useVisionStore = defineStore('vision', () => {
  const resnet = ref<ResnetLoadResult | null>(null)
  const scrfd = ref<{ inputSize: number } | null>(null)
  const busy = ref(false)
  const classification = ref<ResnetClassifyResult | null>(null)
  const detections = ref<ScrfdDetectResult | null>(null)
  const error = ref<string | null>(null)
  const lastImage = ref<string | null>(null)

  async function loadResnet(args: ResnetLoadArgs) {
    busy.value = true
    error.value = null
    try {
      resnet.value = await resnetLoad(args)
    } catch (e) {
      error.value = String(e)
    } finally {
      busy.value = false
    }
  }

  async function classify(file: File, topK = 5) {
    busy.value = true
    error.value = null
    try {
      const buf = new Uint8Array(await file.arrayBuffer())
      lastImage.value = URL.createObjectURL(file)
      classification.value = await resnetClassify(buf, topK)
    } catch (e) {
      error.value = String(e)
    } finally {
      busy.value = false
    }
  }

  async function loadScrfd(modelPath: string, inputSize = 640) {
    busy.value = true
    error.value = null
    try {
      scrfd.value = await scrfdLoad(modelPath, inputSize)
    } catch (e) {
      error.value = String(e)
    } finally {
      busy.value = false
    }
  }

  async function detect(file: File, conf = 0.5) {
    busy.value = true
    error.value = null
    try {
      const buf = new Uint8Array(await file.arrayBuffer())
      lastImage.value = URL.createObjectURL(file)
      detections.value = await scrfdDetect(buf, conf)
    } catch (e) {
      error.value = String(e)
    } finally {
      busy.value = false
    }
  }

  return {
    resnet,
    scrfd,
    busy,
    classification,
    detections,
    error,
    lastImage,
    loadResnet,
    classify,
    loadScrfd,
    detect,
  }
})
